import os
import torch
import pandas as pd
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.models import ResNet34_Weights
from sklearn.preprocessing import LabelEncoder


"""
The purpose of this file is to provide the ability to import functions to other files, which were developed in the 
jupyter notebook. This is required to unpack the exported model into the main.py functions.
"""


decisions_file_path = "./data/decisions.csv"
images_path = "./data/compressed_images"


# Dataset
class AssistantDataset(Dataset):
    def __init__(self, images_dir: str, labels_file: str, transform=None) -> None:
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

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        image_path = os.path.join(self.images_dir, self.labels.iloc[index, 0])
        image = Image.open(image_path).convert('RGB')

        heart_region = image.crop((215, 370, 400,  420))
        in_hand_region = image.crop((200, 410, 600,  450))

        if self.transform:
            image = self.transform(image)
            heart_region = self.transform(heart_region)
            in_hand_region = self.transform(in_hand_region)

        targets = {col: self.label_encoders[col].transform([self.labels.iloc[index][col]])[0]
                   for col in self.label_encoders}

        return image, heart_region, in_hand_region, targets


class HeartsInHandClassifier(nn.Module):
    """
    This model is responsible for detecting in-hand item and player`s hearts.
    """
    def __init__(self, hearts_classes: int, in_hand_classes: int) -> None:
        super(HeartsInHandClassifier, self).__init__()

        self.resnet_hearts = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet_in_hand = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze first 6 layers in order to preserve some basic learned features
        for param in list(self.resnet_hearts.parameters())[:6]:
            param.requires_grad = False
        for param in list(self.resnet_in_hand.parameters())[:6]:
            param.requires_grad = False

        self.resnet_hearts.fc = nn.Identity()
        self.resnet_in_hand.fc = nn.Identity()

        self.fc_hearts = nn.Linear(512, hearts_classes)
        self.fc_in_hand = nn.Linear(512, in_hand_classes)

    def forward(self, heart_region: torch.Tensor, in_hand_region: torch.Tensor) -> dict:
        heart_features = self.resnet_hearts(heart_region)
        in_hand_features = self.resnet_in_hand(in_hand_region)

        hearts_pred = self.fc_hearts(heart_features)
        in_hand_pred = self.fc_in_hand(in_hand_features)

        return {"hearts": hearts_pred, "in_hand_item": in_hand_pred}


class ImageContextClassifier(nn.Module):
    """
    This model responsible for detecting light lvl, player`s activity and target mob.
    """
    def __init__(self, num_classes: dict) -> None:
        super(ImageContextClassifier, self).__init__()
        self.cnn = models.resnet34(weights=ResNet34_Weights.DEFAULT)

        # Freeze first 6 layers in order to preserve some basic learned features
        for param in list(self.cnn.parameters())[:6]:
            param.requires_grad = False
        for param in list(self.cnn.parameters())[6:]:
            param.requires_grad = True

        self.cnn.fc = nn.Identity()

        self.fc = nn.ModuleDict({
            key: nn.Linear(512, num) for key, num in num_classes.items() if key not in ["hearts", "in_hand_item"]
        })

    def forward(self, x: torch.Tensor) -> dict:
        features = self.cnn(x)
        return {key: layer(features) for key, layer in self.fc.items()}


# Ensemble function to combine predictions from both models
def ensemble_predict(
        hearts_model: HeartsInHandClassifier,
        context_model: ImageContextClassifier,
        heart_region: torch.Tensor,
        in_hand_region: torch.Tensor,
        full_image: torch.Tensor
) -> dict:
    hearts_output = hearts_model(heart_region, in_hand_region)
    context_output = context_model(full_image)
    return {**hearts_output, **context_output}


ds = AssistantDataset(images_dir=images_path, labels_file=decisions_file_path)
num_classes = {col: len(ds.label_encoders[col].classes_) for col in ds.label_encoders}
