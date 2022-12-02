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
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch import InceptionResnetV1

import config
from model_tools import get_torch_model, ModelTaskSet
from data_tools import MyDataset, get_labels, get_loaders
from eval_tools import simplified_eval


def train_loop(
    model_path,
    model_func,
    train_loader,
    val_loader,
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0,
    epochs=1,
):
    model = model_func()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()
    avg_vloss = np.inf
    best_vloss = np.inf

    for epoch in range(epochs):
        it = tqdm.tqdm(train_loader)
        running_loss = 0.0
        last_loss = 0.0
        model.train(True)

        for i, (data, target_onehot) in enumerate(it):
            optimizer.zero_grad()
            output = model(data)
            target = torch.argmax(target_onehot, dim=1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)

            # We don't need gradients on to do reporting
            it.set_description(f"{avg_loss=:.4f}")

        model.train(False)
        with torch.no_grad():
            running_vloss = 0.0
            preds = []
            labels = []
            for i, vdata in enumerate(train_loader):
                vinputs, vtarget_onehot = vdata
                vtarget = torch.argmax(vtarget_onehot, dim=1)
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vtarget)
                running_vloss += vloss
                preds.append(voutputs.cpu().numpy().argmax(axis=1))
                labels.append(vtarget.cpu().numpy())
            preds = np.concatenate(preds)
            labels = np.concatenate(labels)

            avg_vloss = running_vloss / (i + 1)
            vscore = simplified_eval(labels, preds)
            print(f"{vscore=} avg_vloss={avg_vloss.item():.4f}")

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(model.state_dict(), f"{model_path}.chk")

    print("Loading the best model.")
    model = model_func()
    model.load_state_dict(torch.load(f"{model_path}.chk"))
    return model


def train(
    model_type="vggface2",
    tasks=("skin_tone", "gender", "age"),
    val_size=1000,
    batch_size=128,
    img_size=160,
    lr=0.001,
    epochs=5,
    seed=1,
    model_path="default_models",
    use_part_of_examples=False,
):
    TRAIN_DATA_PATH = pathlib.Path.cwd() / config.rel_train_data_path
    TEST_DATA_PATH = pathlib.Path.cwd() / config.rel_test_data_path
    MODEL_PATH = pathlib.Path.cwd() / model_path

    torch.manual_seed(seed)
    np.random.seed(seed)

    if not isinstance(tasks, tuple):
        tasks = [tasks]

    full_train_data = pd.read_csv(TRAIN_DATA_PATH / "labels.csv").dropna()
    train_data, val_data = train_test_split(
        full_train_data, test_size=val_size, random_state=seed
    )
    print("Train size:", len(train_data))
    print("Validation size:", len(val_data))

    train_labels_by_task = get_labels(train_data, tasks)
    val_labels_by_task = get_labels(val_data, tasks)
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
    val_loaders = get_loaders(
        TRAIN_DATA_PATH,
        val_data,
        val_labels_by_task,
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
            task=task,
            train_func=train_loop,
            train_loader=train_loaders[task],
            val_loader=val_loaders[task],
            lr=lr,
            epochs=epochs,
        )

    model_set.save()


if __name__ == "__main__":
    fire.Fire(train)
