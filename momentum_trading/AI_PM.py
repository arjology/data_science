import abc
import argparse
from typing import List, Union
from enum import Enum
from tqdm import trange
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Dense

from torch import nn
from torch import optim
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Class to develop your AI portfolio manager
class AIPMDevelopment(abc.ABC):
    def __init__(
        self,
        epochs: int = None,
        loss_fn: Union[str, nn.Module] = None,
        optimizer: Union[str, optim.Optimizer] = None,
        learning_rate: float = None,
        metrics: List[str] = None,
        reduction: str = None,
    ):
        # Read your data in and split the dependent and independent
        data = pd.read_csv("data/IBM.csv")
        X = data["Delta Close"]
        y = data.drop(["Delta Close"], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

        # Params
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.metrics = metrics
        self.reduction = reduction

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass

    @abc.abstractmethod
    def save(self):
        pass


class AIPMDevelopmentTorch(AIPMDevelopment):
    """Pytorch implementation"""

    def __init__(
        self,
        epochs: int = 1000,
        loss_fn: nn.Module = nn.HingeEmbeddingLoss,
        optimizer: optim.Optimizer = optim.RMSprop,
        learning_rate: float = 1e-4,
        reduction: str = "mean",
    ):
        super().__init__(
            epochs=epochs,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learning_rate=learning_rate,
            reduction=reduction,
        )
        print(
            "\n{star}\n{mod}\n{star}".format(
                star="".join(["*"] * 10), mod=f"Using device [{device}]"
            )
        )

        N, D_in, H, D_out = 64, 1, 3, 1
        self.X_train_tensor = (
            torch.tensor(self.X_train.values, dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        self.X_test_tensor = (
            torch.tensor(self.X_test.values, dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        self.y_train_tensor = (
            torch.tensor(self.y_train.values, dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        self.y_test_tensor = (
            torch.tensor(self.y_test.values, dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )

        self.model = nn.Sequential(
            nn.Linear(D_in, D_in),
            nn.Tanh(),
            nn.Linear(D_in, H),
            nn.Tanh(),
            nn.Linear(H, H),
            nn.Tanh(),
            nn.Linear(H, H),
            nn.Tanh(),
            nn.Linear(H, D_out),
            nn.Tanh(),
        ).to(device)

    def train(self):
        loss_fn = self.loss_fn(reduction=self.reduction)
        optimizer = self.optimizer(self.model.parameters())
        with trange(self.epochs) as t:
            for i in t:
                y_pred = self.model(self.X_train_tensor).unsqueeze(1)
                loss = loss_fn(y_pred, self.y_train_tensor)
                if i % 50 == 0:
                    t.set_postfix(epoch=i, loss=loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def evaluate(self):
        # Evaluate the predictions of the model
        y_pred = self.model(self.X_test_tensor).squeeze().detach().numpy()
        y_pred = np.around(y_pred, 0)
        y_test = self.y_test_tensor.squeeze().detach().numpy()
        print(classification_report(y_test, y_pred))

    def save(self):
        torch.save(self.model, "data/torch_model.p")


class AIPMDevelopmentKeras(AIPMDevelopment):
    """Keras implementation"""

    def __init__(
        self,
        epochs: int = 100,
        loss_fn: str = "hinge",
        optimizer: str = "rmsprop",
        metrics: List[str] = ["accuracy"],
    ):
        super().__init__(
            epochs=epochs, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics
        )
        self.network = Sequential()
        self.network.add(Dense(1, input_shape=(1,), activation="tanh"))
        self.network.add(Dense(3, activation="tanh"))
        self.network.add(Dense(3, activation="tanh"))
        self.network.add(Dense(3, activation="tanh"))
        self.network.add(Dense(1, activation="tanh"))

    def train(self,):
        self.network.compile(
            optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics
        )
        self.network.fit(self.X_train.values, self.y_train.values, epochs=self.epochs)

    def evaluate(self):
        # Evaluate the predictions of the model
        y_pred = self.model(self.X_test_tensor).squeeze().detach().numpy()
        y_pred = np.around(y_pred, 0)
        print(classification_report(self.y_test, y_pred))

    def save(self):
        # Save structure to json
        model = self.network.to_json()
        with open("data/keras_model.json", "w") as json_file:
            json_file.write(model)
        # Save weights to HDF5
        self.network.save_weights("data/keras_weights.h5")


class KerasLossFunction(Enum):
    binary_crossentropy = "binary_crossentropy"
    hinge = "hinge"


class PytorchLossFunction(Enum):
    HingeEmbedding = nn.HingeEmbeddingLoss
    MSE = nn.MSELoss
    SoftMargin = nn.SoftMarginLoss


class PortfolioManagementModel:
    def __init__(self):
        # Data in to test that the saving of weights worked
        data = pd.read_csv("./data/IBM.csv")
        X = data["Delta Close"]
        y = data.drop(["Delta Close"], axis=1)
        self.X = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1).to(device)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)

        # Load torch model
        self.model = torch.load("data/torch_model.p")

        # Verify weights and structure are loaded
        y_pred = self.model(self.X).squeeze().detach().numpy()
        y_pred = np.around(y_pred, 0)
        y_test = self.y.squeeze().detach().numpy()
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--framework",
        "-f",
        dest="framework",
        type=str,
        choices=["pytorch", "keras"],
        default="pytorch",
    )
    parser.add_argument("--epochs", "-e", dest="epochs", type=int, default=100)
    parser.add_argument(
        "--reduction",
        "-r",
        dest="reduction",
        type=str,
        default="mean",
        choices=("mean", "sum", "none"),
    )
    parser.add_argument(
        "--loss",
        "-l",
        dest="loss",
        type=str,
        default="HingeEmbeddingLoss",
        choices=("HingeEmbedding", "hinge", "MSE", "binary_crossentropy", "SoftMargin"),
    )
    args = parser.parse_args()
    if args.framework == "pytorch":
        loss_fn = PytorchLossFunction[args.loss].value
        model = AIPMDevelopmentTorch(
            epochs=args.epochs, reduction=args.reduction, loss_fn=loss_fn
        )
    elif args.framework == "keras":
        loss_fn = KerasLossFunction[args.loss].value
        model = AIPMDevelopmentKeras(epochs=args.epochs, loss_fn=loss_fn)
    model.train()
    model.evaluate()
    model.save()
