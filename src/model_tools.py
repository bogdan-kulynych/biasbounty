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

    def fit_task(self, task_name, train_func, *args, **kwargs):
        model_func = self.model_func_by_task[task_name]
        self.models[task_name] = train_func(model_func, *args, **kwargs)

    def save(self):
        if not self.models:
            raise ValueError("No models to save.")

        for model_name, model in self.models.items():
            filename = self.model_path / f"{model_name}.pth"
            print(f"Saving {filename}")
            torch.save(model.state_dict(), filename)

    def load(self):
        for model_file_path in self.model_path.iterdir():
            task = model_file_path.stem
            print(f"Loading {model_file_path}")
            model = self.model_func_by_task[task]()
            model.load_state_dict(torch.load(model_file_path))
            self.models[task] = model

    def predict(self, X):
        preds = [model.predict(X) for model in self.models.values()]
        return preds
