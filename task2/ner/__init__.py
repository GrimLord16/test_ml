"""
ner
===
Transformer-based Named Entity Recognition package for extracting animal
names from free-form text.

Quick start
-----------
>>> from ner.inference import AnimalNerInference
>>> ner = AnimalNerInference("models/ner")
>>> ner.extract("There is a zebra in the picture.")
['zebra']

Modules
-------
dataset
    Synthetic NER data generation and PyTorch Dataset class.
train
    Fine-tuning script (run as ``python -m ner.train --help``).
inference
    Inference wrapper (run as ``python -m ner.inference --help``).
"""

from .inference import AnimalNerInference

__all__ = ["AnimalNerInference"]
