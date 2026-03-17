"""
dataset.py
==========
Synthetic NER dataset generation for animal name extraction.

Generates BIO-tagged sentences using template patterns and a list of
known animal class names from the Mammals-45 dataset.  This synthetic
data is used to fine-tune a transformer-based NER model.

BIO tag scheme
--------------
O         — token is not part of an animal entity
B-ANIMAL  — first token of an animal entity
I-ANIMAL  — continuation token of an animal entity
"""

import random
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


# ---------------------------------------------------------------------------
# Tag vocabulary
# ---------------------------------------------------------------------------

LABEL2ID = {
    "O": 0,
    "B-ANIMAL": 1,
    "I-ANIMAL": 2,
    "B-NEG-ANIMAL": 3,
    "I-NEG-ANIMAL": 4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# Animal class names (Mammals-45 dataset)
# ---------------------------------------------------------------------------

DEFAULT_ANIMALS: List[str] = [
    "antelope", "badger", "bat", "bear", "bison", "boar", "buffalo",
    "cat", "cheetah", "chimpanzee", "cow", "coyote", "deer", "dog",
    "dolphin", "donkey", "elephant", "fox", "giraffe", "goat",
    "gorilla", "hamster", "hare", "hedgehog", "hippopotamus", "horse",
    "hyena", "jaguar", "kangaroo", "koala", "leopard", "lion",
    "lynx", "meerkat", "moose", "mouse", "okapi", "orangutan", "otter",
    "panda", "pig", "porcupine", "raccoon", "rat", "reindeer",
    "rhinoceros", "seal", "sheep", "skunk", "sloth", "squirrel",
    "tiger", "walrus", "weasel", "whale", "wolf", "wombat", "zebra",
    "polar bear", "grizzly bear", "red fox",
]

# Sentence templates — {animal} is replaced with the actual class name
TEMPLATES: List[str] = [
    "There is a {animal} in the picture.",
    "The {animal} looks beautiful.",
    "I can see a {animal} in this image.",
    "This photo shows a {animal}.",
    "A {animal} is visible here.",
    "Look at this {animal}!",
    "Is that a {animal} in the photo?",
    "The picture contains a {animal}.",
    "I think this is a {animal}.",
    "This appears to be a {animal}.",
    "What a magnificent {animal}!",
    "The image depicts a {animal}.",
    "Can you see the {animal} here?",
    "I believe there is a {animal} in this photo.",
    "The {animal} in the image is very clear.",
    "That is definitely a {animal}.",
    "This looks like a {animal} to me.",
    "The photo clearly shows a {animal}.",
    "I spotted a {animal} in this picture.",
    "There seems to be a {animal} here.",
    "The animal I see is a {animal}.",
    "Wow, a {animal}!",
    "In the picture you can spot a {animal}.",
    "My guess is that this is a {animal}.",
    "The subject of this image is a {animal}.",
]

# Negation templates — {animal} is present but negated
NEGATION_TEMPLATES: List[str] = [
    "There is no {animal} in the picture.",
    "I don't see a {animal} here.",
    "This is not a {animal}.",
    "There isn't a {animal} in this image.",
    "No {animal} is visible here.",
    "I can't see any {animal} in this photo.",
    "This doesn't look like a {animal} to me.",
    "The image does not show a {animal}.",
    "There's no {animal} in this photograph.",
    "I don't think this is a {animal}.",
    "This is definitely not a {animal}.",
    "No {animal} appears in this image.",
    "I see no {animal} here.",
    "This photo does not contain a {animal}.",
    "That doesn't look like a {animal} to me.",
    "I wouldn't say this is a {animal}.",
    "The animal here is clearly not a {animal}.",
    "This image contains no {animal}.",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NerExample:
    """A single BIO-tagged sentence split into word tokens and labels."""
    tokens: List[str]
    labels: List[str]  # one BIO label per token, e.g. ["O", "O", "O", "B-ANIMAL", "O"]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _build_labels(tokens: List[str], animal: str, negated: bool = False) -> List[str]:
    """Assign BIO string labels to *tokens* for *animal*, optionally using NEG tags."""
    animal_words = animal.split()
    b_tag = "B-NEG-ANIMAL" if negated else "B-ANIMAL"
    i_tag = "I-NEG-ANIMAL" if negated else "I-ANIMAL"
    labels: List[str] = []
    i = 0
    while i < len(tokens):
        window = [t.lower().rstrip(".,!?") for t in tokens[i : i + len(animal_words)]]
        if window == [w.lower() for w in animal_words]:
            labels.append(b_tag)
            for _ in range(len(animal_words) - 1):
                i += 1
                labels.append(i_tag)
        else:
            labels.append("O")
        i += 1
    return labels


def generate_examples(
    animals: List[str] = DEFAULT_ANIMALS,
    templates: List[str] = TEMPLATES,
    n_per_animal: int = 15,
    seed: int = 42,
    neg_ratio: float = 0.25,
) -> List[NerExample]:
    """
    Generate synthetic NER examples by substituting animal names into templates.

    Parameters
    ----------
    animals : list of str
        Animal class names to use as named entities.
    templates : list of str
        Affirmed sentence templates containing the ``{animal}`` placeholder.
    n_per_animal : int
        Number of sentences generated per animal name.
    seed : int
        Random seed for reproducibility.
    neg_ratio : float
        Fraction of generated sentences that use negation templates (default 0.25).

    Returns
    -------
    list of NerExample
        Shuffled list of labelled sentences.
    """
    random.seed(seed)
    examples: List[NerExample] = []

    for animal in animals:
        n_neg = max(1, int(n_per_animal * neg_ratio))
        n_pos = n_per_animal - n_neg

        for template in random.choices(templates, k=n_pos):
            sentence = template.format(animal=animal)
            tokens = sentence.split()
            examples.append(NerExample(tokens=tokens, labels=_build_labels(tokens, animal, negated=False)))

        for template in random.choices(NEGATION_TEMPLATES, k=n_neg):
            sentence = template.format(animal=animal)
            tokens = sentence.split()
            examples.append(NerExample(tokens=tokens, labels=_build_labels(tokens, animal, negated=True)))

    random.shuffle(examples)
    return examples


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class AnimalNerDataset(Dataset):
    """
    PyTorch Dataset that tokenizes NerExamples for transformer training.

    Handles subword tokenization by assigning ``-100`` to non-first
    subword tokens so they are ignored by ``CrossEntropyLoss``.

    Parameters
    ----------
    examples : list of NerExample
    tokenizer : PreTrainedTokenizerFast
    max_length : int
    """

    def __init__(
        self,
        examples: List[NerExample],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 128,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        encoding = self.tokenizer(
            example.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Align word-level labels to subword tokens.
        # Only the first subword of each word gets the real label;
        # remaining subwords and special tokens get -100 (ignored in loss).
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels: List[int] = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)             # [CLS], [SEP], [PAD]
            elif word_id != prev_word_id:
                aligned_labels.append(LABEL2ID[example.labels[word_id]])
            else:
                aligned_labels.append(-100)             # non-first subword
            prev_word_id = word_id

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(aligned_labels, dtype=torch.long),
        }
