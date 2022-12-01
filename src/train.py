# +
from config import *

# -

# %load_ext autoreload
# %autoreload 2

# +
import os
import json
import pathlib
import functools

import numpy as np
import pandas as pd

import tqdm
import torch
import sklearn

from torch import nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch import InceptionResnetV1

# -

from model_tools import get_torch_model, ModelTaskSet
from data_tools import MyDataset, get_labels, get_loaders

TRAIN_DATA_PATH = pathlib.Path.cwd() / rel_train_data_path
TEST_DATA_PATH = pathlib.Path.cwd() / rel_test_data_path
MODEL_PATH = pathlib.Path.cwd() / rel_model_path

# +
train_data = pd.read_csv(TRAIN_DATA_PATH / "labels.csv").dropna()
# test_data = pd.read_csv(TEST_DATA_PATH / "labels.csv").dropna()

train_labels_by_task = get_labels(train_data, tasks)
# test_labels_by_task = get_labels(test_data, tasks)

num_classes_by_task = {task: train_labels_by_task[task].shape[1] for task in tasks}
print(f"Classes: {num_classes_by_task}")


# +
if do_not_use_all_examples:
    train_data = train_data.iloc[:1000]

train_loaders = get_loaders(
    TRAIN_DATA_PATH,
    train_data,
    train_labels_by_task,
    img_size=img_size,
    batch_size=batch_size,
)
# test_loaders = get_loaders(TEST_DATA_PATH, test_data, test_labels_by_task,
#                            img_size=img_size, batch_size=batch_size)
# -


def train_func(model_func, data_loader):
    model = model_func()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0
    )

    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        it = tqdm.tqdm(enumerate(data_loader))
        for batch_idx, (data, target_onehot) in it:
            output = model(data)
            target = torch.argmax(target_onehot, dim=1)
            loss = criterion(output, target)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                it.set_description(f"{batch_idx=}, {loss.item()=}")

    return model


# +
model_func_by_task = {}
for task in tasks:
    model_func_by_task[task] = functools.partial(
        get_torch_model, model_type=model_type, num_classes=num_classes_by_task[task]
    )


model_set = ModelTaskSet(
    model_path=MODEL_PATH,
    model_func_by_task=model_func_by_task,
)
# -

for task in tasks:
    print(f"Training {task}")
    model_set.fit_task(
        task_name=task, train_func=train_func, data_loader=train_loaders[task]
    )

model_set.save()
