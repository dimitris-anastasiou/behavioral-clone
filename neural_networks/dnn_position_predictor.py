# DNN Position Predictor

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class POSBCRobot(RobotPolicy):

    def __init__(self, seed_value=52):

        # Set seed for reproducibility
        self.set_seed(seed_value)

        # Define the neural network structure
        self.input_size = 4         # Observation size: agent position (2) + goal position (2)
        self.hidden_size = 256
        self.output_size = 4        # Number of actions: up, down, left, right

        # Neural network definition
        self.policy_network = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),                                        # 1st hidden layer
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),                                        # 2nd hidden layer
            nn.Linear(self.hidden_size, self.output_size)
        )

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.1)
        self.criterion = nn.CrossEntropyLoss()

    def set_seed(self, seed_value=42):
        random.seed(seed_value)         # Python random module
        np.random.seed(seed_value)      # Numpy module
        torch.manual_seed(seed_value)   # CPU operations

    def train(self, data):
        for key, val in data.items():
            print(key, val.shape)
        print("Using dummy solution for POSBCRobot")
        pass

        # Unpack the data
        observations = data['obs']
        actions = data['actions']

        # Convert data to tensors
        observations_tensor = torch.tensor(observations, dtype=torch.float)
        actions_tensor = torch.tensor(actions, dtype=torch.long)

        # Training loop
        self.policy_network.train()
        for epoch in range(1200):
            self.optimizer.zero_grad()
            outputs = self.policy_network(observations_tensor)
            loss = self.criterion(outputs, actions_tensor)
            loss.backward()
            self.optimizer.step()

            print(f'Epoch {epoch+1}, Average Loss: {loss.item()}')

            # Check if the loss is low enough to stop training
            if loss.item() < 0.095 :
                print("Stopping training due to low loss.")
                break

        print(f"Training completed with final loss: {loss.item()}")


    def get_action(self, obs):
        # Convert data to tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0)

        # Predict action
        self.policy_network.eval()
        with torch.no_grad():
            output = self.policy_network(obs_tensor)
            predicted_action = torch.argmax(output, dim=1).item()

        return predicted_action
