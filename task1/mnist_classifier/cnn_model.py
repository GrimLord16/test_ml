import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .interface import MnistClassifierInterface

class _ConvNet(nn.Module):
    """
    Two-block convolutional network followed by fully-connected layers.

    Convolutional blocks
    --------------------
    Block 1: Conv2d(1 -> 32, kernel=3, padding=1) + ReLU + MaxPool(2x2)
    Block 2: Conv2d(32 -> 64, kernel=3, padding=1) + ReLU + MaxPool(2x2)

    After two MaxPool operations on a 28x28 input:
        feature-map size = 7x7,  channels = 64  =>  3136 units

    Fully-connected head
    --------------------
    Linear(3136 -> 256) + ReLU + Dropout(0.3) + Linear(256 -> 10)
    """

    def __init__(self, dropout_rate: float = 0.3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 28×28 -> 14×14

            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 14×14 -> 7×7
        )

        # 64 channels × 7 × 7 spatial positions = 3136
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 10), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        x = self.features(x)
        return self.classifier(x)


class CNNMnistClassifier(MnistClassifierInterface):
    """
    MNIST classifier backed by a PyTorch convolutional neural network.

    Parameters
    ----------
    epochs : int, default=10
        Number of full passes through the training set.
    batch_size : int, default=64
        Number of samples per mini-batch.
    lr : float, default=1e-3
        Learning rate for the Adam optimiser.
    device : str or None, default=None
        ``'cuda'``, ``'mps'``, or ``'cpu'``.  Automatically detected when
        ``None``.
    dropout_rate : float, default=0.3
        Dropout probability in the fully-connected head.

    """

    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str | None = None,
        dropout_rate: float = 0.3,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout_rate = dropout_rate

        # Auto-select the best available compute device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.net = _ConvNet(dropout_rate).to(self.device)

    @staticmethod
    def _prepare(X: np.ndarray) -> torch.Tensor:
        """
        Normalise and reshape *X* to (n, 1, 28, 28) float32 tensors.

        Accepts inputs shaped (n, 28, 28) or (n, 784).
        Pixel values are divided by 255 to bring them into [0, 1].
        """
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 2 and X.shape[1] == 784:
            X = X.reshape(-1, 28, 28)               # (n, 784) -> (n, 28, 28)
        elif X.ndim == 3 and X.shape[1:] == (28, 28):
            pass                                     # already (n, 28, 28)
        else:
            raise ValueError(
                f"Expected shape (n, 784) or (n, 28, 28), got {X.shape}."
            )

        X = X / 255.0                                # normalise to [0, 1]
        X = X[:, np.newaxis, :, :]                   # (n, 28, 28) -> (n, 1, 28, 28)
        return torch.tensor(X)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the CNN on the provided data.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, 784) or (n_samples, 28, 28)
            Raw pixel values.
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

            with tqdm(loader, desc=f"[CNN] Epoch {epoch}/{self.epochs}", unit="batch") as bar:
                for X_batch, y_batch in bar:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimiser.zero_grad()
                    logits = self.net(X_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimiser.step()

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
