import torch.nn as nn
import torch.nn.functional as F
import torch


class Network(nn.Module):

    def __init__(self, action_size):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=20, out_channels=32, kernel_size=(3, 3), stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2)

        self.flatten = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(128 * 4 * 4, 512)
        self.fc2a = torch.nn.Linear(512, action_size)
        self.fc2s = torch.nn.Linear(512, 1)

    def forward(self, state):
        x = self.conv1(state)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = F.relu(x)
        action_values = self.fc2a(x)
        state_value = self.fc2s(x)[0]

        return action_values, state_value
