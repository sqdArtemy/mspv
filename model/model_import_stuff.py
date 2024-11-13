import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.preprocessing import LabelEncoder

decisions_file_path = "./data/decisions.csv"
images_path = "./data/compressed_images"


class AssistantDataset(Dataset):
    def __init__(self, images_dir: str, labels_file: str, transform=None):
        self.images_dir = images_dir
        self.labels = pd.read_csv(labels_file)
        self.transform = transform or transforms.ToTensor()

        self.label_encoders = {
            "activity": LabelEncoder(),
            "hearts": LabelEncoder(),
            "light_lvl": LabelEncoder(),
            "in_hand_item": LabelEncoder(),
            "target_mob": LabelEncoder(),
        }

        for col in self.label_encoders:
            self.label_encoders[col].fit(self.labels[col])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.labels.iloc[index, 1])
        image = Image.open(image_path).convert('RGB')

        targets = {col: self.label_encoders[col].transform([self.labels.iloc[index][col]])[0]
                   for col in self.label_encoders}

        if self.transform:
            image = self.transform(image)

        return image, targets


class AssistantClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AssistantClassifier, self).__init__()
        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in list(self.cnn.parameters())[:6]:
            param.requires_grad = True
        self.cnn.fc = nn.Identity()

        self.fc = nn.ModuleDict({
            key: nn.Linear(512, num) for key, num in num_classes.items()
        })

    def forward(self, x):
        features = self.cnn(x)
        return {key: layer(features) for key, layer in self.fc.items()}


ds = AssistantDataset(images_dir=images_path, labels_file=decisions_file_path)
num_classes = {col: len(ds.label_encoders[col].classes_) for col in ds.label_encoders}
