# CNN Navigation Agent - Multi Map

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

class RGBBCRobot2(RobotPolicy):
    def __init__(self, seed_value=42):
        super().__init__()

        # Set seed for reproducibility
        self.set_seed(seed_value)

        # Initialize CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        self.optimizer = optim.Adam(self.cnn.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def set_seed(self, seed_value=42):
        random.seed(seed_value)         # Python random module
        np.random.seed(seed_value)      # Numpy module
        torch.manual_seed(seed_value)   # CPU operations

    def train(self, data):

        for key, val in data.items():
            print(key, val.shape)
        print("Using dummy solution for RGBBCRobot2")
        pass

        # Unpack the data and convert to tensors
        observations = torch.tensor(data['obs'], dtype=torch.float32).permute(0, 3, 1, 2)
        actions = torch.tensor(data['actions'], dtype=torch.long)

        dataset = TensorDataset(observations, actions)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Training loop
        self.cnn.train()
        for epoch in range(20):
            epoch_loss = 0.0
            for batch in data_loader:
                inputs, labels = batch
                self.optimizer.zero_grad()
                outputs = self.cnn(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f'Epoch {epoch+1}, Average Loss: {epoch_loss / len(data_loader)}')

            # Check if the loss is low enough to stop training
            if epoch_loss / len(data_loader) < 0.14:
                print("Stopping training due to low loss.")
                break

    def get_action(self, obs):

        # Convert data to tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)

        self.cnn.eval()
        # Predict action
        with torch.no_grad():
            output = self.cnn(obs_tensor)
            predicted_action = torch.argmax(output, dim=1).item()

        return predicted_action
