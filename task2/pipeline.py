"""
Usage (CLI):
    python pipeline.py --text "There is a dog here." --image dog.jpg

    # Custom model paths:
    python pipeline.py \\
        --text "I can see a cat in this picture." \\
        --image cat.jpg \\
        --ner_model_dir ./ner/ner_model \\
        --classifier_model_dir ./image_classification/animal_classifier

    # Note: train the models first:
    #   cd ner && python train.py --output_dir ./ner_model
    #   cd image_classification && python train.py --data_dir ./mammals45 --output_dir ./animal_classifier
"""

import argparse
import importlib.util
import os
import sys


def _load_module(name, file_path):
    """Load a module from an absolute file path under a given module name."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base = os.path.dirname(__file__)
_ner_inference = _load_module(
    "ner_inference", os.path.join(_base, "ner", "inference.py")
)
_cls_inference = _load_module(
    "cls_inference", os.path.join(_base, "image_classification", "inference.py")
)

AnimalNER = _ner_inference.AnimalNER
AnimalClassifier = _cls_inference.AnimalClassifier


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

# Minimal plural → singular rules for Mammals-45 class names.
PLURAL_MAP = {
    # Irregular plurals
    "mice": "mouse",
    "wolves": "wolf",
    "hippopotami": "hippopotamus",
    "rhinoceroses": "rhinoceros",
    "rhinoceri": "rhinoceros",
    "chimpanzees": "chimpanzee",
    "orangutans": "orangutan",
    # Generic suffix rules handled by _normalize_animal()
}


def _normalize_animal(name: str) -> str:
    """
    Normalize an animal name to a canonical singular lower-case form.

    Handles:
    - Explicit irregular plural forms (from PLURAL_MAP)
    - Common English plural suffix rules (-ies → -y, -es → strip, -s → strip)
    """
    if name is None:
        return ""
    name = name.strip().lower().replace("_", " ")

    if name in PLURAL_MAP:
        return PLURAL_MAP[name]

    if name.endswith("ies") and len(name) > 4:
        return name[:-3] + "y"
    if name.endswith("ses") or name.endswith("xes") or name.endswith("zes"):
        return name[:-2]
    if name.endswith("es") and len(name) > 3:
        candidate = name[:-2]
        if len(candidate) >= 3:
            return candidate
    if name.endswith("s") and not name.endswith("ss") and len(name) > 3:
        return name[:-1]          # dogs → dog, cats → cat

    return name


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class AnimalVerificationPipeline:
    """
    End-to-end pipeline that answers: "Does the animal in the text match the
    animal in the image?"

    Parameters
    ----------
    ner_model_dir : str
        Path to the fine-tuned NER model directory (from ner/train.py).
    classifier_model_dir : str
        Path to the fine-tuned image classifier directory (from
        image_classification/train.py).
    """

    def __init__(self, ner_model_dir: str, classifier_model_dir: str):
        print("Initializing Animal Verification Pipeline...")
        self.ner = AnimalNER(model_dir=ner_model_dir)
        self.classifier = AnimalClassifier(model_dir=classifier_model_dir)
        print("Pipeline ready.\n")

    def run(self, text: str, image_path: str) -> bool:
        """
        Verify that the animal mentioned in *text* matches the animal in *image_path*.

        Parameters
        ----------
        text : str
            Natural language description, e.g. "There is a dog in the picture."
        image_path : str
            Path to the image file.

        Returns
        -------
        bool
            - Affirmed animal in text  → True if image matches it.
            - Negated animal in text   → True if image does NOT show it.
            - No animal mentioned      → True (text makes no claim).
              Note: the image classifier always predicts one of 45 animal
              classes, so "no animal in image" cannot be detected without
              a separate confidence threshold.
        """
        affirmed = [_normalize_animal(a) for a in self.ner.extract_animals(text)]
        negated  = [_normalize_animal(a) for a in self.ner.extract_negated_animals(text)]
        print(f"Affirmed animals (text): {affirmed}")
        print(f"Negated  animals (text): {negated}")

        raw_animal_image = self.classifier.predict(image_path)
        animal_image = _normalize_animal(raw_animal_image)
        print(f"Predicted animal (image): {raw_animal_image!r}  →  normalized: {animal_image!r}")

        if affirmed:
            # Text asserts an animal is present — image must match it
            match = animal_image in affirmed
        elif negated:
            # Text asserts an animal is absent — image must NOT show it
            match = animal_image not in negated
        else:
            # Text makes no animal claim — nothing to contradict
            match = True

        print(f"Match: {match}")
        return match


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify whether the animal described in a text matches the "
            "animal shown in an image."
        )
    )
    parser.add_argument(
        "--text",
        required=True,
        help='Text description, e.g. "There is a dog in the picture."',
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image file to classify",
    )
    parser.add_argument(
        "--ner_model_dir",
        default="./ner/ner_model",
        help="Path to the NER model directory (default: ./ner/ner_model)",
    )
    parser.add_argument(
        "--classifier_model_dir",
        default="./image_classification/animal_classifier",
        help=(
            "Path to the image classifier model directory "
            "(default: ./image_classification/animal_classifier)"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = AnimalVerificationPipeline(
        ner_model_dir=args.ner_model_dir,
        classifier_model_dir=args.classifier_model_dir,
    )

    print(f'Input text : "{args.text}"')
    print(f"Input image: {args.image}\n")

    result = pipeline.run(text=args.text, image_path=args.image)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
