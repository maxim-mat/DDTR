import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import tensor
import torch.nn.functional as F

np.random.seed(0)


def replace_values(x):
    # Replace -1 with 1 in the last column
    last_column = x[:, -1]
    last_column = torch.where(last_column == -1, torch.tensor(1), last_column)
    x[:, -1] = last_column

    # Replace -1 with 0
    x = torch.where(x == -1, torch.tensor(0), x)

    return x


def preprocess(data):
    dim_padding = [F.pad(tensor(d), (0, 1, 0, 0), mode='constant', value=0) for d in data]
    i, max_length_sequence = max(enumerate(dim_padding), key=lambda y: len(y[1]))  # this will dictate the final shape of tensor
    rem = len(max_length_sequence) % 8
    if rem != 0:
        for _ in range(8 - rem):
            max_length_sequence = torch.vstack([max_length_sequence, max_length_sequence[-1]])
        dim_padding[i] = max_length_sequence
    t = pad_sequence(dim_padding, batch_first=True, padding_value=-1)
    return torch.stack([replace_values(ti) for ti in t])


class TraceDataset(Dataset):
    def __init__(self, labels, data=None):
        if data is not None and len(data) != len(labels):
            raise ValueError("data and labels must be of same length")
        self.__data = preprocess(data) if data is not None else None
        self.__labels = preprocess(labels)
        self.sequence_length = self.__data.shape[1]

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return self.__labels[idx], self.__data[idx] if self.__data is not None else None
