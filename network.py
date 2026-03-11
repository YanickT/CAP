from typing import List, Union, Iterable
import torch
import time


class SpinNetwork:
    """
    Basis class for the network which tries to determine the spin
    """

    def __init__(self, device: Union[str, torch.device] = "cpu"):
        """
        Initialize the network
        :param device: Union[str, torch.device] = device to do the calculations on
        """
        self.device = device
        self.loss = torch.nn.L1Loss()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(128 ** 2, 128 ** 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128 ** 2, 64 ** 2),
            torch.nn.Tanh(),
            torch.nn.Linear(64 ** 2, 64 ** 2),
            torch.nn.Tanh(),
            torch.nn.Linear(64 ** 2, 32 ** 2),
            torch.nn.Tanh(),
            torch.nn.Linear(32 ** 2, 1),
            torch.nn.Tanh()
        )
        self.model.to(self.device)
        self.optimizer = None

    def train(self, train_data: Iterable, test_data: Iterable, its: int = 5, lr: float = 5e-3) -> List[float]:
        """
        Train and Test the created model
        :param train_data: Iterable = set of trainings data
        :param test_data: Iterable = set of test data
        :param its: int = number of training epochs
        :param lr: float = learning rate for Adam
        :return: List[float] = Loss for each trained epoch (determined by test data)
        """
        losses = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        for i in range(its):
            self.model.train()
            t1 = time.time()
            for inp, out in train_data:
                loss = self.loss(self.model(inp), out)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            loss_epoch = self.eval(test_data)
            losses.append(sum(loss_epoch) / len(loss_epoch))

            print(f"Epoche:{i}\nTime: {time.time() - t1}s\nLoss: {losses[-1]}\n\n")

        return losses

    def eval(self, test_data: Iterable) -> List[float]:
        """
        Evaluate the created network
        :param test_data: Iterable = set of test data
        :return: List[float] = Loss for given set
        """
        self.model.eval()
        with torch.no_grad():
            loss_epoch = []
            for inp, out in test_data:
                res = self.model(inp)
                loss_epoch.append(self.loss(res, out).item())

        return loss_epoch

    def predict(self, image: torch.tensor) -> float:
        """
        Predict the spin for a given image
        :param image: torch.tensor[1d] = image of the BH
        :return: float = predicted spin
        """
        self.model.eval()
        with torch.no_grad():
            res = self.model(image).item()
        return res


class IncNetwork:
    """
    Basis class for the network which tries to determine the inclination
    """

    def __init__(self, device="cpu"):
        """
        Initialize the network
        :param device: Union[str, torch.device] = device to do the calculations on
        """
        self.device = device
        self.loss = torch.nn.MSELoss()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(128 ** 2, 128 ** 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128 ** 2, 64 ** 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64 ** 2, 64 ** 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64 ** 2, 32 ** 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32 ** 2, 1),
            torch.nn.LeakyReLU()
        )
        self.model.to(self.device)
        self.optimizer = None

    def train(self, train_data: Iterable, test_data: Iterable, its: int = 5, lr: float = 5e-3) -> List[float]:
        """
        Train and Test the created model
        :param train_data: Iterable = set of trainings data
        :param test_data: Iterable = set of test data
        :param its: int = number of training epochs
        :param lr: float = learning rate for Adam
        :return: List[float] = Loss for each trained epoch (determined by test data)
        """
        losses = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        for i in range(its):
            self.model.train()
            t1 = time.time()
            for inp, out in train_data:
                loss = self.loss(self.model(inp), out)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            loss_epoch = self.eval(test_data)
            losses.append(sum(loss_epoch) / len(loss_epoch))

            print(f"Epoche:{i}\nTime: {time.time() - t1}s\nLoss: {losses[-1]}\n\n")

        return losses

    def eval(self, test_data: Iterable) -> List[float]:
        """
        Evaluate the created network
        :param test_data: Iterable = set of test data
        :return: List[float] = Loss for given set
        """
        self.model.eval()
        with torch.no_grad():
            loss_epoch = []
            for inp, out in test_data:
                res = self.model(inp)
                loss_epoch.append(self.loss(res, out).item())

        return loss_epoch

    def predict(self, image: torch.tensor) -> float:
        """
        Predict the spin for a given image
        :param image: torch.tensor[1d] = image of the BH
        :return: float = predicted spin
        """
        self.model.eval()
        with torch.no_grad():
            res = self.model(image).item()
        return res
