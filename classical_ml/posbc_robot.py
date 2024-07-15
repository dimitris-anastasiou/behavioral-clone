# POSBCRobot

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class POSBCRobot():
    def train(self, data):
        """
        A method for training a policy.

        Args:
            data: a dictionary that contains X (observations) and y (actions).

        Returns:
            This method does not return anything. It only need to update the
            property of a RobotPolicy instance.
        """
        # Initialize empty lists to store features and labels
        Xobservations = []
        Yactions = []

        # Iterate through each entry in the dataset
        Xobservations = data['obs']
        Yactions = data['actions'].reshape(-1)

        # Convert lists to numpy arrays for easier manipulation and to fit with many machine learning libraries
        Xobservations_array = np.array(Xobservations)
        Yactions_array = np.array(Yactions)

        # Train the model
        self.clf = KNeighborsClassifier(n_neighbors=50, weights='uniform')
        self.clf.fit(Xobservations_array, Yactions_array)
        pass

    def get_actions(self, observations):
        """
        A method for getting actions. You can do data preprocessing and feed
        forward of your trained model here.

        Args:
            observations: a batch of observations (images or vectors)

        Returns:
            A batch of actions with the same batch size as observations.
        """
        observations_array = np.array(observations)

        # Predict actions
        Yactions_predicted = self.clf.predict(observations_array)
        return Yactions_predicted
