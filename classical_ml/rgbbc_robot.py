# RGBBCRobot

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.image import extract_patches_2d

class RGBBCRobot():

    def __init__(self, n_clusters=50, patch_size=(8, 8), max_patches_per_image=100):
        self.n_clusters = n_clusters
        self.patch_size = patch_size
        self.max_patches_per_image = max_patches_per_image
        self.scaler = StandardScaler()

    def extract_patches_and_create_vocabulary(self, observations):
        # Extract patches
        patches = [extract_patches_2d(obs.reshape(64, 64, 3), self.patch_size, max_patches=self.max_patches_per_image).reshape(-1, self.patch_size[0] * self.patch_size[1] * 3) for obs in observations]
        patches = np.vstack(patches)

        # Fit k-means to create the vocabulary
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(patches)

    def image_to_feature_histogram(self, image):
        # Extract patches from the image
        patches = extract_patches_2d(image.reshape(64, 64, 3), self.patch_size, max_patches=self.max_patches_per_image).reshape(-1, self.patch_size[0] * self.patch_size[1] * 3)

        # Predict cluster assignments for each patch
        if patches.shape[0] > 0:
            words = self.kmeans.predict(patches)
            # Build a histogram of word occurrences
            histogram, _ = np.histogram(words, bins=range(self.n_clusters + 1), density=True)
        else:
            histogram = np.zeros(self.n_clusters)
        return histogram

    def train(self, data):
        """
        A method for training a policy.

        Args:
            data: a dictionary that contains X (observations) and y (actions).

        Returns:
            This method does not return anything. It will just need to update the
            property of a RobotPolicy instance.
        """
       # Extract patches and create the vocabulary
        self.extract_patches_and_create_vocabulary(data['obs'])

        # Convert images to histograms of visual word occurrences
        X = np.array([self.image_to_feature_histogram(obs) for obs in data['obs']])

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train the model
        Y = data['actions'].reshape(-1)
        self.clf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, max_depth=None, max_features=None)
        self.clf.fit(X_scaled, Y)
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
        # Convert images to histograms of visual word occurrences
        X = np.array([self.image_to_feature_histogram(obs) for obs in observations])

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict actions
        Yactions_predicted = self.clf.predict(X_scaled)
        return Yactions_predicted
