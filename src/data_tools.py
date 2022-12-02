import pathlib

import torchvision

import torch
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer

from PIL import Image
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1, 2, 0))
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


def get_labels(data, tasks):
    binarizers = {task: LabelBinarizer() for task in tasks}
    label_dict = {}
    for task in tasks:
        lab = binarizers[task].fit_transform(data[task])
        if lab.shape[1] == 1:
            label_dict[task] = np.hstack((1 - lab, lab))
        else:
            label_dict[task] = lab
    return label_dict


def get_task_loader(data_path, data, task_labels, img_size, batch_size):
    length = width = 64  # size for each input image, increase if you want
    num_examples = data.shape[0]
    X = []
    y = []
    for i in range(num_examples):
        img_ = torchvision.io.read_image(
            str(pathlib.Path(data_path) / data.iloc[i]["name"])
        )
        if img_.shape[0] == 3:
            X.append(img_.numpy())
            y.append(task_labels[i])

    X = np.array(X)
    y = np.array(y)
    # print(f"{X.shape=} {y.shape=}")
    print("X.shape={} y.shape={}".format(X.shape, y.shape))

    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    dataset = MyDataset(X, y, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    return data_loader


def get_loaders(data_path, data, labels, *args, **kwargs):
    loaders = {}
    for task, task_labels in labels.items():
        # print(f"Creating data loader for {task}")
        print("Creating data loader for {}".format(task))
        loaders[task] = get_task_loader(data_path, data, task_labels, *args, **kwargs)
    return loaders
