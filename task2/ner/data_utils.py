"""
data_utils.py - Synthetic NER Training Data Generation for Animal Entity Recognition

Generates tokenized sentences with BIO tags for fine-tuning a token classification model
to extract animal names from natural language text.

The primary ANIMALS list covers the 45 mammal classes from the Mammals-45 dataset
plus common synonyms and additional animals to improve generalization.
"""

import random
from datasets import Dataset


# ---------------------------------------------------------------------------
# Animal vocabulary (40+ animals for generalization beyond the 10 target classes)
# ---------------------------------------------------------------------------
ANIMALS = [
    # Mammals-45 dataset classes (45 mammal species)
    "antelope", "badger", "bat", "bear", "bison", "boar", "buffalo",
    "cat", "cheetah", "chimpanzee", "cow", "coyote", "deer", "dog",
    "dolphin", "donkey", "elephant", "fox", "giraffe", "goat",
    "gorilla", "hamster", "hare", "hedgehog", "hippopotamus", "horse",
    "hyena", "jaguar", "kangaroo", "koala", "leopard", "lion",
    "lynx", "meerkat", "moose", "mouse", "okapi", "orangutan", "otter",
    "panda", "pig", "porcupine", "raccoon", "rat", "rhinoceros",
    "seal", "sheep", "skunk", "sloth", "squirrel", "tiger",
    "walrus", "weasel", "whale", "wolf", "wombat", "zebra",
    # Multi-word variants (tests I-ANIMAL tag handling)
    "polar bear", "grizzly bear", "brown bear", "black bear",
    "red fox", "grey wolf", "snow leopard", "giant panda",
]

# ---------------------------------------------------------------------------
# Sentence templates — {animal} is replaced with the animal name at generation time
# ---------------------------------------------------------------------------
SENTENCE_TEMPLATES = [
    "There is a {animal} in the picture.",
    "I can see a {animal} here.",
    "The image shows a {animal}.",
    "A {animal} is visible in this photo.",
    "Look at that {animal}!",
    "This picture contains a {animal}.",
    "Can you spot the {animal} in the image?",
    "There appears to be a {animal} in the photograph.",
    "The photo features a {animal}.",
    "I see a {animal} in the background.",
    "A beautiful {animal} is shown in the image.",
    "The picture depicts a {animal} in the wild.",
    "Is that a {animal} I see?",
    "What a lovely {animal} in this picture!",
    "The image clearly shows a {animal}.",
    "There's definitely a {animal} in this photo.",
    "This photograph captures a {animal}.",
    "I believe there is a {animal} in this image.",
    "The subject of this photo appears to be a {animal}.",
    "A {animal} can be found in this picture.",
    "Someone photographed a {animal} here.",
    "The wildlife photo shows a {animal}.",
    "This looks like a {animal} to me.",
    "The animal in the image is a {animal}.",
    "Can you identify the {animal} in this photo?",
    "There's a {animal} somewhere in this image.",
    "A stunning {animal} appears in this shot.",
    "The camera captured a {animal} in this scene.",
    "You can clearly see a {animal} in this picture.",
    "This nature photo includes a {animal}.",
    "I found a {animal} in the image.",
    "The {animal} is the main subject of this photo.",
    "A wild {animal} was photographed here.",
    "This is a photo of a {animal}.",
    "Notice the {animal} in the corner of the image.",
]

# ---------------------------------------------------------------------------
# BIO label constants
# ---------------------------------------------------------------------------
LABEL_O = 0             # Outside any entity
LABEL_B_ANIMAL = 1      # Beginning of an affirmed ANIMAL entity
LABEL_I_ANIMAL = 2      # Inside (continuation of) an affirmed ANIMAL entity
LABEL_B_NEG_ANIMAL = 3  # Beginning of a negated ANIMAL entity
LABEL_I_NEG_ANIMAL = 4  # Inside (continuation of) a negated ANIMAL entity

LABEL_NAMES = ["O", "B-ANIMAL", "I-ANIMAL", "B-NEG-ANIMAL", "I-NEG-ANIMAL"]


# ---------------------------------------------------------------------------
# Negation sentence templates — {animal} is present but negated
# ---------------------------------------------------------------------------
NEGATION_TEMPLATES = [
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
# Core helpers
# ---------------------------------------------------------------------------

def _tokenize_text(text: str) -> list[str]:
    """
    Naive whitespace + punctuation tokenizer that mirrors what BIO taggers
    typically operate on before sub-word tokenization.

    Returns a list of word tokens (punctuation is split off as its own token).
    """
    import re
    # Split on whitespace; also isolate trailing punctuation
    tokens = []
    for raw in text.split():
        # Separate trailing punctuation (.,!?) from the word
        match = re.match(r"^([\w'-]+)([.,!?]*)$", raw)
        if match:
            word, punct = match.group(1), match.group(2)
            tokens.append(word)
            if punct:
                tokens.append(punct)
        else:
            tokens.append(raw)
    return tokens


def _build_bio_tags(tokens: list[str], animal: str, negated: bool = False) -> list[int]:
    """
    Assign BIO tags to *tokens* given that *animal* is the entity to label.

    The animal name may contain multiple space-separated words (e.g. "blue whale"),
    which receive B-ANIMAL + I-ANIMAL tags respectively.
    When *negated* is True, B-NEG-ANIMAL / I-NEG-ANIMAL are used instead.

    The search is case-insensitive. Returns a label list aligned with *tokens*.
    """
    animal_words = animal.lower().split()
    n_animal = len(animal_words)
    labels = [LABEL_O] * len(tokens)
    b_tag = LABEL_B_NEG_ANIMAL if negated else LABEL_B_ANIMAL
    i_tag = LABEL_I_NEG_ANIMAL if negated else LABEL_I_ANIMAL

    i = 0
    while i < len(tokens):
        if (tokens[i].lower() == animal_words[0] and
                i + n_animal <= len(tokens) and
                [t.lower() for t in tokens[i:i + n_animal]] == animal_words):
            labels[i] = b_tag
            for j in range(1, n_animal):
                labels[i + j] = i_tag
            i += n_animal
        else:
            i += 1

    return labels


def _generate_sample(template: str, animal: str, negated: bool = False) -> dict:
    """
    Fill *template* with *animal*, tokenize, and produce BIO labels.

    Returns a dict with keys: tokens, ner_tags.
    """
    sentence = template.replace("{animal}", animal)
    tokens = _tokenize_text(sentence)
    ner_tags = _build_bio_tags(tokens, animal, negated=negated)
    return {"tokens": tokens, "ner_tags": ner_tags}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_ner_dataset(
    n_samples: int = 2000,
    seed: int = 42,
    neg_ratio: float = 0.25,
) -> Dataset:
    """
    Generate a synthetic NER dataset for animal entity recognition.

    Parameters
    ----------
    n_samples : int
        Total number of sentences to generate (default 2000).
    seed : int
        Random seed for reproducibility.
    neg_ratio : float
        Fraction of sentences that contain a *negated* animal mention (default 0.25).

    Returns
    -------
    datasets.Dataset
        HuggingFace Dataset with columns: tokens (list[str]), ner_tags (list[int]).
    """
    random.seed(seed)

    n_neg = int(n_samples * neg_ratio)
    n_pos = n_samples - n_neg

    samples = []
    for _ in range(n_pos):
        template = random.choice(SENTENCE_TEMPLATES)
        animal = random.choice(ANIMALS)
        samples.append(_generate_sample(template, animal, negated=False))

    for _ in range(n_neg):
        template = random.choice(NEGATION_TEMPLATES)
        animal = random.choice(ANIMALS)
        samples.append(_generate_sample(template, animal, negated=True))

    random.shuffle(samples)
    dataset = Dataset.from_list(samples)
    return dataset


def tokenize_and_align_labels(examples: dict, tokenizer) -> dict:
    """
    Tokenize word-level tokens and align BIO labels to sub-word tokens.

    This function is designed to be used with `Dataset.map()`.  For each
    word token the first sub-word receives the original label; continuation
    sub-words receive -100 so they are ignored by the CrossEntropy loss.

    Parameters
    ----------
    examples : dict
        Batch from a HuggingFace Dataset containing 'tokens' and 'ner_tags'.
    tokenizer : PreTrainedTokenizerFast
        A fast HuggingFace tokenizer (required for word_ids()).

    Returns
    -------
    dict
        Tokenizer output dict augmented with an 'labels' key.
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,  # input is already word-tokenized
        padding="max_length",
        max_length=128,
    )

    all_labels = []
    for batch_idx, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=batch_idx)
        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                # Special tokens ([CLS], [SEP], padding) → ignore
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                # First sub-word of a new word → assign the word's label
                aligned_labels.append(word_labels[word_id])
            else:
                # Continuation sub-word → ignore in loss
                aligned_labels.append(-100)
            prev_word_id = word_id
        all_labels.append(aligned_labels)

    tokenized["labels"] = all_labels
    return tokenized


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating 10 sample sentences...\n")
    ds = generate_ner_dataset(n_samples=10, seed=0)
    for row in ds:
        tagged = list(zip(row["tokens"], [LABEL_NAMES[t] for t in row["ner_tags"]]))
        print(tagged)
    print(f"\nLabel scheme: {LABEL_NAMES}")
