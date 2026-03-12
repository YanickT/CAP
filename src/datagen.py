from typing import List, Union, Tuple
import torch
import numpy as np
import re

"""
functions to extract:
-> Correct spin
-> Correct inclination
from the filename
Example filename:
-> a-0.00_inc010_kv0.20_h1.00_GRRT.npz
"""
functions = {
    "a": (lambda x: float(re.findall("a-?\d+.?\d*", x)[0].replace("a", ""))),
    "inc": (lambda x: float(re.findall("inc\d+", x)[0].replace("inc", "")))
}


class DataGen:
    """
    Iterable for the train and test data
    The data are loaded not all at once to save RAM
    """

    def __init__(self, files: List[str], target: str, bs: int = 64, path: str = "",
                 device: Union[str, torch.device] = "cpu"):
        """
        Initialize a data Iterator
        :param files: List[str] = List of filenames with the image data
        :param target: str = either 'a' or 'inc' for spin or inclination, respectively
        :param bs: int = batch size
        :param path: str = path to the files
        :param device: Union[str, torch.device] =  device to put the tensors on
        """
        self.f = functions[target]
        self.bs = bs
        self.path = path
        self.device = device

        self.files = files
        self.length = len(self.files)

        self.images = torch.empty((self.bs, 128 ** 2))
        self.targets = torch.empty((self.bs, 1))

    def __iter__(self) -> Tuple[torch.tensor, torch.tensor]:
        """
        Iterate over the DataGen object
        :return: yield Tuple[torch.tensor, torch.tensor] = (input_data, target_data)
        """
        c = 0
        for file in self.files:
            if c == self.bs:
                yield self.images.to(self.device), self.targets.to(self.device)
                c = 0

            self.images[c, :] = torch.from_numpy(np.load(f"{self.path}/{file}")["image"].flatten()).float()
            self.targets[c] = self.f(file)
            c += 1

        # necessary since set may not be exactly dividable by bs
        # in this case some data will occur twice (filling the tensor to bs)
        yield self.images.to(self.device), self.targets.to(self.device)
