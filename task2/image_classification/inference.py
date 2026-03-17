"""
inference.py - Animal Image Classification Inference

Loads a fine-tuned ResNet-18 model and classifies an animal image.
Works with any number of classes — the class list is read from
class_mapping.json produced by train.py.

Usage (CLI):
    python inference.py --model_dir ./animal_classifier --image_path bear.jpg
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


# ---------------------------------------------------------------------------
# ImageNet normalization (must match training)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_IMG_SIZE = 224


# ---------------------------------------------------------------------------
# AnimalClassifier class
# ---------------------------------------------------------------------------

class AnimalClassifier:
    """
    Inference wrapper for the trained ResNet-18 animal classifier.

    Parameters
    ----------
    model_dir : str
        Path to the directory containing:
          - best_model.pth     (model weights saved by train.py)
          - class_mapping.json (index → English class name)
    """

    def __init__(self, model_dir: str):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Model directory not found: '{model_dir}'. "
                "Run train.py first to generate the model."
            )

        # Load class mapping: {"0": "dog", "1": "horse", ...}
        mapping_path = os.path.join(model_dir, "class_mapping.json")
        if not os.path.isfile(mapping_path):
            raise FileNotFoundError(
                f"class_mapping.json not found in '{model_dir}'."
            )
        with open(mapping_path) as f:
            raw_mapping = json.load(f)
        # Keys may be strings (JSON) — convert to int
        self.idx_to_class: dict[int, str] = {int(k): v for k, v in raw_mapping.items()}
        self.num_classes = len(self.idx_to_class)

        # Build and load the model
        print(f"Loading classifier model from '{model_dir}'...")
        self.model = self._build_model(self.num_classes)
        weights_path = os.path.join(model_dir, "best_model.pth")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"best_model.pth not found in '{model_dir}'."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Inference-time transform (no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        print(f"Classifier loaded ({self.num_classes} classes, device: {self.device}).")



    @staticmethod
    def _build_model(num_classes: int) -> torch.nn.Module:
        """Rebuild the same ResNet-18 architecture used during training."""
        model = models.resnet18(weights=None)  # no pretrained weights needed here
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        return model

    def _preprocess(self, image_path: str) -> torch.Tensor:
        """Load an image from disk and apply the inference transform."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: '{image_path}'")
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)          # (C, H, W)
        return tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)


    def predict(self, image_path: str) -> str:
        """
        Classify an image and return the predicted English class name.

        Parameters
        ----------
        image_path : str
            Path to the image file (JPEG, PNG, etc.).

        Returns
        -------
        str
            Predicted animal name, e.g. "dog".
        """
        tensor = self._preprocess(image_path)
        with torch.no_grad():
            logits = self.model(tensor)          # (1, num_classes)
        pred_idx = logits.argmax(dim=1).item()
        return self.idx_to_class[pred_idx]

    def predict_proba(self, image_path: str) -> dict[str, float]:
        """
        Classify an image and return per-class probabilities.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        dict[str, float]
            Mapping from class name to predicted probability, sorted descending.
        """
        tensor = self._preprocess(image_path)
        with torch.no_grad():
            logits = self.model(tensor)          # (1, num_classes)
        probs = F.softmax(logits, dim=1)[0].tolist()  # list of floats
        prob_dict = {self.idx_to_class[i]: round(probs[i], 6)
                     for i in range(len(probs))}
        # Sort by probability descending for readability
        return dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify an animal image using a fine-tuned ResNet-18"
    )
    parser.add_argument(
        "--model_dir",
        default="./animal_classifier",
        help="Path to saved classifier model directory (default: ./animal_classifier)",
    )
    parser.add_argument(
        "--image_path",
        required=True,
        help="Path to the image file to classify",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    classifier = AnimalClassifier(model_dir=args.model_dir)

    predicted = classifier.predict(args.image_path)
    probabilities = classifier.predict_proba(args.image_path)

    print(f"\nImage       : {args.image_path}")
    print(f"Prediction  : {predicted}")
    print("\nTop-5 class probabilities:")
    for i, (cls, prob) in enumerate(list(probabilities.items())[:5], start=1):
        print(f"  {i}. {cls:12s} {prob:.4f}")


if __name__ == "__main__":
    main()
