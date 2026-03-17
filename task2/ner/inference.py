"""
inference.py - Animal NER Inference

Loads a fine-tuned DistilBERT token classifier and extracts animal names from text.

Usage (CLI):
    python inference.py --model_dir ./ner_model --text "There is a dog in the picture."
"""

import argparse
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


# ---------------------------------------------------------------------------
# Label constants (must match training configuration)
# ---------------------------------------------------------------------------
LABEL_NAMES = ["O", "B-ANIMAL", "I-ANIMAL", "B-NEG-ANIMAL", "I-NEG-ANIMAL"]
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}


class AnimalNER:
    """
    Named Entity Recognizer for animal mentions in natural language text.

    Parameters
    ----------
    model_dir : str
        Path to the directory containing the saved fine-tuned model
        and tokenizer (produced by train.py).
    """

    def __init__(self, model_dir: str):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Model directory not found: '{model_dir}'. "
                "Run train.py first to generate the model."
            )

        print(f"Loading NER model from '{model_dir}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"NER model loaded (device: {self.device}).")

    # Internal helpers

    def _predict_labels(self, text: str) -> list[tuple[str, str]]:
        """
        Run the model on *text* and return a list of (word, label) pairs.

        The word-level labels are recovered by taking the label of the first
        sub-word token for each original word.
        """
        # Tokenize keeping track of word offsets
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
        logits = outputs.logits  # (1, seq_len, num_labels)
        predictions = torch.argmax(logits, dim=-1)[0].tolist()  # (seq_len,)

        # Map sub-word predictions back to words
        word_ids = encoding["input_ids"][0]  # kept for length reference
        # Use the fast tokenizer's word_ids feature
        encoding_cpu = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        # word_ids() is available on the BatchEncoding object (not the dict sent to model)
        word_id_list = encoding_cpu.word_ids(batch_index=0)

        # Split original text into words for reconstruction
        words = text.split()
        word_to_label: dict[int, str] = {}

        for token_idx, word_id in enumerate(word_id_list):
            if word_id is None:
                continue  # special token
            if word_id not in word_to_label:
                # First sub-word of this word → assign its label
                word_to_label[word_id] = ID2LABEL[predictions[token_idx]]

        word_label_pairs = []
        for idx, word in enumerate(words):
            label = word_to_label.get(idx, "O")
            word_label_pairs.append((word, label))

        return word_label_pairs

    def _collect_spans(self, pairs: list[tuple[str, str]], b_tag: str, i_tag: str) -> list[str]:
        """Reconstruct entity spans from word-label pairs for the given B/I tag pair."""
        entities: list[str] = []
        current: list[str] = []

        for word, label in pairs:
            if label == b_tag:
                if current:
                    entities.append(" ".join(current))
                current = [word]
            elif label == i_tag and current:
                current.append(word)
            else:
                if current:
                    entities.append(" ".join(current))
                    current = []

        if current:
            entities.append(" ".join(current))

        return [e.rstrip(".,!?;:").lower() for e in entities if e.rstrip(".,!?;:")]

    def extract_animals(self, text: str) -> list[str]:
        """
        Extract all *affirmed* animal entity spans from *text*.

        Negated mentions (e.g. "there is no bear") are recognised by the model
        and silently discarded — use :meth:`extract_negated_animals` to retrieve them.

        Parameters
        ----------
        text : str
            Input sentence, e.g. "There is a blue whale in the image."

        Returns
        -------
        list[str]
            List of extracted animal names (may be empty).
        """
        pairs = self._predict_labels(text)
        return self._collect_spans(pairs, "B-ANIMAL", "I-ANIMAL")

    def extract_negated_animals(self, text: str) -> list[str]:
        """
        Extract animal names that are explicitly *negated* in *text*.

        Parameters
        ----------
        text : str
            Input sentence, e.g. "There is no bear in this photo."

        Returns
        -------
        list[str]
            List of negated animal names (may be empty).
        """
        pairs = self._predict_labels(text)
        return self._collect_spans(pairs, "B-NEG-ANIMAL", "I-NEG-ANIMAL")

    def predict(self, text: str) -> str | None:
        """
        Return the first extracted animal name from *text*, or None if none found.

        Parameters
        ----------
        text : str
            Input sentence.

        Returns
        -------
        str | None
            First detected animal name (lower-cased) or None.
        """
        animals = self.extract_animals(text)
        return animals[0] if animals else None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract animal names from text using a fine-tuned NER model"
    )
    parser.add_argument(
        "--model_dir",
        default="./ner_model",
        help="Path to the saved NER model directory (default: ./ner_model)",
    )
    parser.add_argument(
        "--text",
        required=True,
        help='Input text, e.g. "There is a dog in the picture."',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ner = AnimalNER(model_dir=args.model_dir)

    print(f'\nInput text : "{args.text}"')

    animals = ner.extract_animals(args.text)
    negated = ner.extract_negated_animals(args.text)
    first = ner.predict(args.text)

    print(f"Affirmed animals  : {animals}")
    print(f"Negated animals   : {negated}")
    print(f"Primary prediction: {first}")


if __name__ == "__main__":
    main()
