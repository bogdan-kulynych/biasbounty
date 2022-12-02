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
SUBMISSION_PATH = pathlib.Path.cwd() / rel_submission_path

# +
train_data = pd.read_csv(TRAIN_DATA_PATH / "labels.csv").dropna()
test_data = pd.read_csv(TEST_DATA_PATH / "labels.csv").dropna()

train_labels_by_task = get_labels(train_data, tasks)
test_labels_by_task = get_labels(test_data, tasks)

num_classes_by_task = {task: train_labels_by_task[task].shape[1] for task in tasks}
print(f"Classes: {num_classes_by_task}")


# +
if do_not_use_all_examples:
    test_data = test_data.iloc[:30]
    for task in tasks:
        test_labels_by_task[task] = test_labels_by_task[task][:30]

test_loaders = get_loaders(
    TEST_DATA_PATH,
    test_data,
    test_labels_by_task,
    img_size=img_size,
    batch_size=batch_size,
)

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

model_set.load()

# -
preds = model_set.predict(test_loaders)


# calculate accuracy
acc = {}
for task in tasks:
    acc[task] = accuracy_score(test_labels_by_task[task].argmax(axis=1), preds[task])


# calculate disparity
def disparity_score(ytrue, ypred):
    cm = confusion_matrix(ytrue, ypred)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    all_acc = list(cm.diagonal())
    return max(all_acc) - min(all_acc)


disp = {}
for task in tasks:
    disp[task] = disparity_score(test_labels_by_task[task].argmax(axis=1), preds[task])

results = {"accuracy": acc, "disparity": disp}


def get_score(results):
    acc = results["accuracy"]
    disp = results["disparity"]
    return (
        2 * acc["gender"] * (1 - disp["gender"])
        + 4 * acc["age"] * (1 - disp["age"] ** 2)
        + 10 * acc["skin_tone"] * (1 - disp["skin_tone"] ** 5)
    )


submission = {
    "submission_name": submission_title,
    "score": get_score(results),
    "metrics": results,
}

print(submission)

with open(SUBMISSION_PATH, "w+") as f:
    json.dump(submission, f)
