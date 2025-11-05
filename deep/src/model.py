import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyFaceCNN(nn.Module):
    """
    Input: (B,1,36,36)
    Output: logits for 2 classes (neg, pos)
    Designed so we can later convert final FC to conv for dense sliding-window.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # 18x18
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # 9x9
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(3)                       # 3x3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*3*3, 128), nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def make_model():
    return TinyFaceCNN()
