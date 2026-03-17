# Task 1 — MNIST Image Classification with OOP

A clean, extensible Python package that wraps three different MNIST classifiers
behind a **single unified API** using object-oriented design principles.

---

## Project Structure

```
task1/
├── mnist_classifier/          # Python package
│   ├── __init__.py            # Public exports
│   ├── interface.py           # Abstract base class (MnistClassifierInterface)
│   ├── rf_model.py            # Random Forest implementation
│   ├── nn_model.py            # Feed-forward Neural Network implementation
│   ├── cnn_model.py           # Convolutional Neural Network implementation
│   └── mnist_classifier.py    # Unified facade (MnistClassifier)
├── demo.ipynb                 # End-to-end demonstration notebook
├── requirements.txt
└── README.md
```

---

## Setup

```bash

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Import the package

```python
import sys
sys.path.insert(0, ".")          # run from the task1/ directory

from mnist_classifier import MnistClassifier
```

### Load MNIST data

```python
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np

# Download once; cached afterwards
train_ds = MNIST(root="./data", train=True,  download=True)
test_ds  = MNIST(root="./data", train=False, download=True)

X_train = train_ds.data.numpy()        # (60000, 28, 28)  uint8
y_train = train_ds.targets.numpy()     # (60000,)
X_test  = test_ds.data.numpy()         # (10000, 28, 28)  uint8
y_test  = test_ds.targets.numpy()      # (10000,)
```

### Random Forest

```python
from sklearn.metrics import accuracy_score

clf_rf = MnistClassifier(algorithm='rf', n_estimators=200)
clf_rf.train(X_train, y_train)
preds_rf = clf_rf.predict(X_test)
print(f"RF accuracy: {accuracy_score(y_test, preds_rf):.4f}")

# Save / load
clf_rf.save("rf_model.pkl")
clf_rf.load("rf_model.pkl")
```

### Feed-Forward Neural Network

```python
clf_nn = MnistClassifier(algorithm='nn', epochs=10, batch_size=64, lr=1e-3)
clf_nn.train(X_train, y_train)
preds_nn = clf_nn.predict(X_test)
print(f"NN accuracy: {accuracy_score(y_test, preds_nn):.4f}")

clf_nn.save("nn_model.pt")
clf_nn.load("nn_model.pt")
```

### Convolutional Neural Network

```python
clf_cnn = MnistClassifier(algorithm='cnn', epochs=10, batch_size=64, lr=1e-3)
clf_cnn.train(X_train, y_train)
preds_cnn = clf_cnn.predict(X_test)
print(f"CNN accuracy: {accuracy_score(y_test, preds_cnn):.4f}")

clf_cnn.save("cnn_model.pt")
clf_cnn.load("cnn_model.pt")
```

### Custom kwargs

All keyword arguments beyond `algorithm` are forwarded to the underlying
classifier constructor:

```python
# Tune the Random Forest
clf_rf = MnistClassifier(algorithm='rf', n_estimators=500, max_depth=20)

# Train the NN for longer
clf_nn = MnistClassifier(algorithm='nn', epochs=30, lr=5e-4)
```

---

## OOP Design

### Abstract Interface (`MnistClassifierInterface`)

Defined in `interface.py` using Python's `abc.ABC`.  Two abstract methods
enforce the contract every classifier must fulfil:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `train` | `(X_train, y_train) -> None` | Fit the model in-place |
| `predict` | `(X) -> np.ndarray` | Return predicted labels |

### Facade (`MnistClassifier`)

`MnistClassifier` is the single public entry point.  It:

1. Accepts an `algorithm` string (`'rf'`, `'nn'`, `'cnn'`).
2. Instantiates the appropriate concrete class.
3. Delegates `train` and `predict` calls to the inner classifier.
4. Provides `save` / `load` using the serialisation format best suited to
   each back-end (pickle for RF; `torch.save` for neural networks).

This **Strategy + Facade** pattern means callers never need to import or
instantiate the concrete classes directly; swapping algorithms is a
one-character change.

---

## Model Architectures

### Random Forest (`rf`)

- Library: `sklearn.ensemble.RandomForestClassifier`
- Pre-processing: flatten 28×28 images → 784-element vectors (no
  normalisation needed for tree-based methods)
- Default hyper-parameters: 200 trees, unlimited depth, all CPU cores

### Feed-Forward NN (`nn`)

```
Input (784)
  → Linear(784→512) → ReLU → Dropout(0.3)
  → Linear(512→256) → ReLU → Dropout(0.3)
  → Linear(256→128) → ReLU → Dropout(0.2)
  → Linear(128→10)
```

- Library: PyTorch
- Pre-processing: flatten + divide by 255 → [0, 1]
- Optimiser: Adam, lr=1e-3
- Loss: CrossEntropyLoss
- Training: 10 epochs, batch size 64, tqdm progress bars

### CNN (`cnn`)

```
Input (1, 28, 28)
  → Conv2d(1→32, 3×3, pad=1) → ReLU → MaxPool(2×2)   → (32, 14, 14)
  → Conv2d(32→64, 3×3, pad=1) → ReLU → MaxPool(2×2)  → (64, 7, 7)
  → Flatten (3136)
  → Linear(3136→256) → ReLU → Dropout(0.3)
  → Linear(256→10)
```

- Library: PyTorch
- Pre-processing: reshape to (n, 1, 28, 28) + divide by 255 → [0, 1]
- Optimiser: Adam, lr=1e-3
- Loss: CrossEntropyLoss
- Training: 10 epochs, batch size 64, tqdm progress bars

---

## Expected Accuracy (test set, defaults)

| Algorithm | Typical accuracy |
|-----------|-----------------|
| RF        | ~97%            |
| NN        | ~98%            |
| CNN       | ~99%            |

---

## Running the Notebook

```bash
cd task1
jupyter notebook demo.ipynb
```
