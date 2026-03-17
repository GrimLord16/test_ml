import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .interface import MnistClassifierInterface


class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    MNIST classifier backed by a scikit-learn Random Forest.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of trees in the forest.  More trees generally improve
        accuracy at the cost of slower training and inference.
    max_depth : int or None, default=None
        Maximum depth of each tree.  ``None`` means trees are expanded
        until all leaves are pure (or contain fewer than
        ``min_samples_split`` samples).
    n_jobs : int, default=-1
        Number of parallel jobs for training/prediction.
        ``-1`` uses all available CPU cores.
    random_state : int, default=42
        Seed for reproducibility.
    **rf_kwargs
        Any additional keyword arguments are forwarded directly to
        ``sklearn.ensemble.RandomForestClassifier``.

    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        n_jobs: int = -1,
        random_state: int = 42,
        **rf_kwargs,
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
            **rf_kwargs,
        )

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        """
        Ensure *X* is a 2-D array of shape (n_samples, 784).

        Handles both already-flat (n, 784) and image-shaped (n, 28, 28)
        inputs.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            # (n_samples, 28, 28) -> (n_samples, 784)
            return X.reshape(X.shape[0], -1)
        if X.ndim == 2 and X.shape[1] == 784:
            return X
        raise ValueError(
            f"Expected X with shape (n, 784) or (n, 28, 28), got {X.shape}."
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the Random Forest on the training data.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, 784) or (n_samples, 28, 28)
            Raw pixel values (any numeric dtype).
        y_train : np.ndarray, shape (n_samples,)
            Integer labels 0-9.
        """
        X_flat = self._flatten(X_train)
        self.model.fit(X_flat, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict digit labels for the input images.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 784) or (n_samples, 28, 28)
            Raw pixel values.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted labels (integers 0-9).
        """
        X_flat = self._flatten(X)
        return self.model.predict(X_flat)
