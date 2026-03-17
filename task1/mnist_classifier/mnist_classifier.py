import pickle
from pathlib import Path

import numpy as np
import torch

from .interface import MnistClassifierInterface
from .rf_model import RandomForestMnistClassifier
from .nn_model import NNMnistClassifier
from .cnn_model import CNNMnistClassifier


_ALGORITHM_MAP: dict[str, type] = {
    "rf": RandomForestMnistClassifier,
    "nn": NNMnistClassifier,
    "cnn": CNNMnistClassifier,
}

class MnistClassifier:
    """
    Unified MNIST classifier facade.

    Wraps one of three back-ends – Random Forest, Feed-Forward NN, or CNN –
    behind a single, consistent API.

    Parameters
    ----------
    algorithm : {'rf', 'nn', 'cnn'}
        Which classifier to use:

        * ``'rf'``  – :class:`RandomForestMnistClassifier`
        * ``'nn'``  – :class:`NNMnistClassifier` (feed-forward network)
        * ``'cnn'`` – :class:`CNNMnistClassifier` (convolutional network)

    **kwargs
        Additional keyword arguments forwarded verbatim to the chosen
        classifier's constructor (e.g. ``n_estimators=300`` for RF or
        ``epochs=20`` for NN/CNN).

    Raises
    ------
    ValueError
        If *algorithm* is not one of the supported values.

    Examples
    --------
    >>> clf = MnistClassifier(algorithm='rf', n_estimators=100)
    >>> clf.train(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    >>> clf.save('my_rf_model.pkl')
    """

    def __init__(self, algorithm: str, **kwargs) -> None:
        algorithm = algorithm.lower().strip()
        if algorithm not in _ALGORITHM_MAP:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Choose from: {sorted(_ALGORITHM_MAP.keys())}."
            )
        self.algorithm = algorithm
        self._classifier: MnistClassifierInterface = _ALGORITHM_MAP[algorithm](**kwargs)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._classifier.train(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._classifier.predict(X)

    def save(self, path: str | Path) -> None:
        """
        Persist the trained model to disk.

        * RF models are saved with :mod:`pickle` (``*.pkl``).
        * NN/CNN models are saved with :func:`torch.save`.

        Parameters
        ----------
        path : str or Path
            Destination file path.  The parent directory must already exist.
        """
        path = Path(path)
        if self.algorithm == "rf":
            with path.open("wb") as fh:
                pickle.dump(self._classifier, fh, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            torch.save(self._classifier, path)
        print(f"Model saved to '{path}'.")

    def load(self, path: str | Path) -> None:
        """
        Restore a previously saved model from disk.

        The *algorithm* attribute of this ``MnistClassifier`` instance must
        match the algorithm used when the model was saved.

        Parameters
        ----------
        path : str or Path
            Source file path produced by :meth:`save`.
\
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No model file found at '{path}'.")

        if self.algorithm == "rf":
            with path.open("rb") as fh:
                self._classifier = pickle.load(fh)
        else:
            self._classifier = torch.load(path, weights_only=False)
        print(f"Model loaded from '{path}'.")

    def __repr__(self) -> str:
        return (
            f"MnistClassifier(algorithm='{self.algorithm}', "
            f"classifier={self._classifier.__class__.__name__})"
        )
