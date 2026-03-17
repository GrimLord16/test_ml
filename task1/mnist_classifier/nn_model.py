import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .interface import MnistClassifierInterface


class _FeedForwardNet(nn.Module):
    """
    Four-layer fully-connected network for 10-class digit classification.

    Parameters
    ----------
    dropout_rate_1 : float
        Dropout probability applied after the first and second hidden layers.
    dropout_rate_2 : float
        Dropout probability applied after the third hidden layer.
    """

    def __init__(self, dropout_rate_1: float = 0.3, dropout_rate_2: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1: 784 -> 512
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate_1),
            # Layer 2: 512 -> 256
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate_1),
            # Layer 3: 256 -> 128
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate_2),
            # Output layer: 128 -> 10
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class NNMnistClassifier(MnistClassifierInterface):
    """
    MNIST classifier backed by a PyTorch feed-forward neural network.

    Parameters
    ----------
    epochs : int, default=10
        Number of full passes through the training data.
    batch_size : int, default=64
        Number of samples per mini-batch.
    lr : float, default=1e-3
        Learning rate for the Adam optimiser.
    device : str or None, default=None
        ``'cuda'``, ``'mps'``, or ``'cpu'``.  When ``None`` the best
        available device is selected automatically.
    dropout_rate_1 : float, default=0.3
        Dropout probability for the first two hidden layers.
    dropout_rate_2 : float, default=0.2
        Dropout probability for the third hidden layer.
    """

    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str | None = None,
        dropout_rate_1: float = 0.3,
        dropout_rate_2: float = 0.2,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout_rate_1 = dropout_rate_1
        self.dropout_rate_2 = dropout_rate_2

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.net = _FeedForwardNet(dropout_rate_1, dropout_rate_2).to(self.device)

    @staticmethod
    def _prepare(X: np.ndarray) -> torch.Tensor:
        """
        Flatten and normalise *X* then return a float32 CPU tensor.

        Pixels are divided by 255 to bring values into [0, 1].
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)  # (n, 28, 28) -> (n, 784)
        elif X.ndim != 2 or X.shape[1] != 784:
            raise ValueError(
                f"Expected shape (n, 784) or (n, 28, 28), got {X.shape}."
            )
        X = X / 255.0  # normalise to [0, 1]
        return torch.tensor(X)


    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the feed-forward network.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, 784) or (n_samples, 28, 28)
            Raw uint8 or float pixel values.
        y_train : np.ndarray, shape (n_samples,)
            Integer labels 0-9.
        """
        X_tensor = self._prepare(X_train)
        y_tensor = torch.tensor(np.asarray(y_train, dtype=np.int64))

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.net.train()
        for epoch in range(1, self.epochs + 1):
            running_loss = 0.0
            correct = 0
            total = 0

            with tqdm(loader, desc=f"[NN] Epoch {epoch}/{self.epochs}", unit="batch") as bar:
                for X_batch, y_batch in bar:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimiser.zero_grad()
                    logits = self.net(X_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimiser.step()

                    # Accumulate statistics for the progress bar
                    running_loss += loss.item() * X_batch.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += X_batch.size(0)

                    bar.set_postfix(
                        loss=f"{running_loss / total:.4f}",
                        acc=f"{correct / total:.4f}",
                    )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict digit labels for the input images.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 784) or (n_samples, 28, 28)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted labels (integers 0-9).
        """
        X_tensor = self._prepare(X).to(self.device)

        self.net.eval()
        with torch.no_grad():
            logits = self.net(X_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()

        return preds
