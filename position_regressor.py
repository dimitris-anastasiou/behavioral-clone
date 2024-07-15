# Position Regressor

import numpy as np
from sklearn import linear_model

class PositionRegressor():

    def train(self, data):
        """
        A method that trains a regressor using the given data

        Args:
            data: a dictionary that contains images and the corresponding ground
            truth location of an agent.

        Returns:
            Nothing
        """
        # Initialize empty lists to store features and labels
        observations = []
        agent_pos = []

        # Iterate through each entry in the dataset
        observations = data['obs']
        flattened_observations = np.array([obs.flatten() for obs in observations])
        agent_pos = [entry["agent_pos"] for entry in data["info"]]

        # Convert lists to numpy arrays for easier manipulation and to fit with many machine learning libraries
        observations_array = np.array(flattened_observations)
        agent_pos_array = np.array(agent_pos)

        # Train the model
        self.reg = linear_model.LinearRegression()
        self.reg.fit(flattened_observations, agent_pos_array)
        pass

    def predict(self, Xs):
        """
        A method that predicts y's given a batch of X's

        Args:
            Xs: a batch of data (in this project, it is in the shape [batch_size, 64, 64, 3])

        Returns:
            The predicted locations (y's) of the agent from your trained model. Note that
            this method expects batched inputs and returns batched outputs
        """
        flattened_Xs = np.array([x.flatten() for x in Xs])
        Ys = self.reg.predict(flattened_Xs)
        return Ys
