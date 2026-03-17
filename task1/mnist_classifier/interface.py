from abc import ABC, abstractmethod
import numpy as np


class MnistClassifierInterface(ABC):
    """
    Abstract base class for MNIST digit classifiers.

    All classifiers in this package follow this contract:
      - ``train``   fits the model on labelled training data.
      - ``predict`` returns class predictions for new samples.

    Subclasses are free to add algorithm-specific hyper-parameters and
    helper methods, but they **must** override both abstract methods.
    """

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the classifier on the provided training data.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, 28, 28) or (n_samples, 784)
            Pixel values of the training images.  Values are expected in the
            range [0, 255] (uint8 or float).  Concrete implementations are
            responsible for any reshaping or normalisation they require.
        y_train : np.ndarray, shape (n_samples,)
            Integer class labels in the range [0, 9].

        Returns
        -------
        None
            The model is updated in-place.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the digit class for each sample in *X*.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 28, 28) or (n_samples, 784)
            Pixel values of the images to classify.  The same preprocessing
            that was applied during ``train`` must be applied here.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted integer class labels in the range [0, 9].
        """
