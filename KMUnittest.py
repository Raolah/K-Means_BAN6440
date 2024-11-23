import unittest
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


class TestKMeansClustering(unittest.TestCase):

    def setUp(self):
        # Load the dataset
        self.file_path = r'C:\Users\Public\PycharmProjects\PythonProject\data\us_daily.csv'
        self.data = pd.read_csv(self.file_path)

        # Select relevant features for clustering
        self.features = self.data[['positiveIncrease', 'negativeIncrease', 'deathIncrease', 'hospitalizedIncrease']]

        # Handle missing values by filling them with the mean of the column
        self.features = self.features.fillna(self.features.mean())

        # Perform K-Means clustering
        self.kmeans = KMeans(n_clusters=3, random_state=0).fit(self.features)

        # Add the cluster labels to the original dataset
        self.data['Cluster'] = self.kmeans.labels_

    def test_data_loading(self):
        # Test if the data is loaded correctly
        self.assertFalse(self.data.empty, "The dataset should not be empty.")
        self.assertIn('positiveIncrease', self.data.columns, "The dataset should contain 'positiveIncrease' column.")
        self.assertIn('negativeIncrease', self.data.columns, "The dataset should contain 'negativeIncrease' column.")
        self.assertIn('deathIncrease', self.data.columns, "The dataset should contain 'deathIncrease' column.")
        self.assertIn('hospitalizedIncrease', self.data.columns,
                      "The dataset should contain 'hospitalizedIncrease' column.")

    def test_missing_values_handling(self):
        # Test if missing values are handled correctly
        self.assertFalse(self.features.isnull().values.any(), "There should be no missing values in the features.")

    def test_kmeans_clustering(self):
        # Test if K-Means clustering is performed correctly
        self.assertEqual(len(self.kmeans.cluster_centers_), 3, "There should be 3 cluster centers.")
        self.assertEqual(len(self.data['Cluster'].unique()), 3, "There should be 3 unique clusters.")

    def test_cluster_labels(self):
        # Test if cluster labels are added to the dataset
        self.assertIn('Cluster', self.data.columns, "The dataset should contain 'Cluster' column.")
        self.assertEqual(len(self.data['Cluster'].unique()), 3, "There should be 3 unique cluster labels.")


if __name__ == '__main__':
    unittest.main()