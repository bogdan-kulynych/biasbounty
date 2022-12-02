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

from model_tools import get_torch_model, ModelTaskSet
from data_tools import MyDataset, get_labels, get_loaders
from eval_tools import get_score, disparity_score

import config


def eval_tasks(
    model_type="vggface2",
    model_path="default_models",
    tasks=("skin_tone", "gender", "age"),
    img_size=160,
    batch_size=128,
    use_part_of_examples=False,
):
    TRAIN_DATA_PATH = pathlib.Path.cwd() / config.rel_train_data_path
    TEST_DATA_PATH = pathlib.Path.cwd() / config.rel_test_data_path
    SUBMISSION_PATH = pathlib.Path.cwd() / config.rel_submission_path
    MODEL_PATH = pathlib.Path.cwd() / model_path

    if not isinstance(tasks, tuple):
        tasks = [tasks]

    train_data = pd.read_csv(TRAIN_DATA_PATH / "labels.csv").dropna()
    test_data = pd.read_csv(TEST_DATA_PATH / "labels.csv").dropna()

    train_labels_by_task = get_labels(train_data, tasks)
    test_labels_by_task = get_labels(test_data, tasks)

    num_classes_by_task = {task: train_labels_by_task[task].shape[1] for task in tasks}
    print(f"Classes: {num_classes_by_task}")

    if use_part_of_examples:
        test_data = test_data.iloc[:100]
        for task in tasks:
            test_labels_by_task[task] = test_labels_by_task[task][:100]

    test_loaders = get_loaders(
        TEST_DATA_PATH,
        test_data,
        test_labels_by_task,
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

    model_set = ModelTaskSet(
        model_path=MODEL_PATH,
        model_func_by_task=model_func_by_task,
    )
    print(model_set.model_func_by_task)

    model_set.load()

    preds = model_set.predict(test_loaders)

    # calculate accuracy
    acc = {}
    for task in tasks:
        acc[task] = accuracy_score(
            test_labels_by_task[task].argmax(axis=1), preds[task]
        )

    disp = {}
    for task in tasks:
        disp[task] = disparity_score(
            test_labels_by_task[task].argmax(axis=1), preds[task]
        )

    results = {"accuracy": acc, "disparity": disp}
    submission = {
        "submission_name": config.submission_title,
        "score": get_score(results),
        "metrics": results,
    }
    print(submission)

    with open(SUBMISSION_PATH, "w+") as f:
        json.dump(submission, f)


if __name__ == "__main__":
    fire.Fire(eval_tasks)
