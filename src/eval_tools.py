import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def disparity_score(ytrue, ypred):
    cm = confusion_matrix(ytrue, ypred)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    all_acc = list(cm.diagonal())
    return max(all_acc) - min(all_acc)


def simplified_eval(ytrue, ypred):
    acc = accuracy_score(ytrue, ypred)
    disp = disparity_score(ytrue, ypred)
    disp = 0.5
    return acc * (1 - disp)


def get_score(results):
    acc = results["accuracy"]
    disp = results["disparity"]

    try:
        score = (
            2 * acc["gender"] * (1 - disp["gender"])
            + 4 * acc["age"] * (1 - disp["age"] ** 2)
            + 10 * acc["skin_tone"] * (1 - disp["skin_tone"] ** 5)
        )
    except KeyError:
        raise ValueError("Not all metrics are available.")
