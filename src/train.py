# +
from config import *

import os
import json
import pathlib
import functools

import numpy as np
import pandas as pd

import fire
import tqdm
import torch
import sklearn

from torch import nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch import InceptionResnetV1

import config
from model_tools import get_torch_model, ModelTaskSet
from data_tools import MyDataset, get_labels, get_loaders


def train_func(
    model_func, data_loader, lr=0.001, momentum=0.9, weight_decay=0.0, epochs=1
):
    print(lr, momentum, weight_decay, epochs)
    model = model_func()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=momentum, weight_decay=weight_decay
    )

    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        it = tqdm.tqdm(data_loader)
        for batch_idx, (data, target_onehot) in enumerate(it):
            output = model(data)
            target = torch.argmax(target_onehot, dim=1)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                it.set_description(
                    "batch_idx={}, loss={}".format(batch_idx, loss.item())
                )

    return model


def train(
    model_type="vggface2",
    tasks=("skin_tone", "gender", "age"),
    img_size=160,
    lr=0.001,
    batch_size=128,
    epochs=5,
    model_path="default_models",
    use_part_of_examples=False,
):
    print()
    TRAIN_DATA_PATH = pathlib.Path.cwd() / config.rel_train_data_path
    TEST_DATA_PATH = pathlib.Path.cwd() / config.rel_test_data_path
    MODEL_PATH = pathlib.Path.cwd() / model_path
    if not isinstance(tasks, tuple):
        tasks = [tasks]

    train_data = pd.read_csv(TRAIN_DATA_PATH / "labels.csv").dropna()
    train_labels_by_task = get_labels(train_data, tasks)

    num_classes_by_task = {task: train_labels_by_task[task].shape[1] for task in tasks}
    print(f"Classes: {num_classes_by_task}")

    if use_part_of_examples:
        train_data = train_data.iloc[:100]

    train_loaders = get_loaders(
        TRAIN_DATA_PATH,
        train_data,
        train_labels_by_task,
        img_size=img_size,
        batch_size=batch_size,
    )

    model_func_by_task = {}
    for task in tasks:
        model_func_by_task[task] = functools.partial(
            get_torch_model,
            model_type=model_type,
            num_classes=num_classes_by_task[task],
        )

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    model_set = ModelTaskSet(
        model_path=MODEL_PATH,
        model_func_by_task=model_func_by_task,
    )

    for task in tasks:
        print(f"Training {task}")
        model_set.fit_task(
            task_name=task,
            train_func=train_func,
            data_loader=train_loaders[task],
            lr=lr,
            epochs=epochs,
        )

    model_set.save()


if __name__ == "__main__":
    fire.Fire(train)
