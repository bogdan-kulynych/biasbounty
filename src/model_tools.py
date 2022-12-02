import numpy as np

import pathlib
import torch


def get_torch_model(model_type, num_classes):
    if model_type == "dummy":
        from torch import nn

        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(128, num_classes, bias=True),
        )

    elif model_type == "vggface2":
        from facenet_pytorch import InceptionResnetV1

        return InceptionResnetV1(
            pretrained="vggface2", classify=True, num_classes=num_classes
        )


class ModelTaskSet:
    def __init__(self, model_func_by_task, model_path):
        self.model_func_by_task = model_func_by_task
        self.model_path = pathlib.Path(model_path)
        self.models = {}

    def fit_task(self, task, train_func, *args, **kwargs):
        model_func = self.model_func_by_task[task]
        model_path = self.model_path / f"{task}.pth"
        self.models[task] = train_func(model_path, model_func, *args, **kwargs)

    def save(self):
        if not self.models:
            raise ValueError("No models to save.")

        for task, model in self.models.items():
            filename = self.model_path / f"{task}.pth"
            print(f"Saving {filename}")
            torch.save(model.state_dict(), filename)

    def load(self):
        for model_file_path in self.model_path.iterdir():
            print(model_file_path)
            ext = model_file_path.suffix
            if ext != ".pth":
                continue
            task = model_file_path.stem
            if task not in self.model_func_by_task:
                continue
            print(f"Loading {task} from {model_file_path}")
            model = self.model_func_by_task[task]()
            model.load_state_dict(torch.load(model_file_path))
            self.models[task] = model

    def predict(self, data_loaders):
        preds_by_task = {}
        for task, data_loader in data_loaders.items():
            preds_by_task[task] = []
            for x, y in data_loader:
                batch_preds = self.models[task](x).detach().numpy()
                preds_by_task[task].append(batch_preds.argmax(axis=1))
            preds_by_task[task] = np.concatenate(preds_by_task[task])

        return preds_by_task
