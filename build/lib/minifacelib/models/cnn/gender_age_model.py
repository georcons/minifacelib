import torch
import torch.nn as nn


class GenderAgeModel(nn.Module):
    def __init__(self, num_genders: int = 2):
        super().__init__()

        # Backbone (shared to age and gender)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)

        # Heads
        self.gender_head = nn.Linear(128, num_genders)  # classification
        self.age_head = nn.Linear(128, 1)  # regression

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.flatten(1)
        x = torch.relu(self.fc1(x))

        gender_logits = self.gender_head(x)
        age = self.age_head(x).squeeze(1)

        return gender_logits, age
