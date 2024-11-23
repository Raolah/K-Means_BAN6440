# K-Means Clustering Application on us_daily.csv Dataset

**Overview**
This application does K-Means clustering on the us_daily.csv dataset, containing COVID-19 statistics daily in the United States. The aim is to cluster the data into insightful groups based on a selected set of features that could give an insight into the patterns and trends of the pandemic.

**Features**
Data Loading: This loads the dataset using pandas.
Feature Selection: To select features of interest for clustering.
Handling Missing Values: Fill in the missing values with the mean across columns.
K-Means Clustering: Perform K-Means clustering on the data into 3 clusters. 
Adding Cluster Labels: Add the cluster labels to the original dataset.
Visualization: Visualize the clusters with matplotlib.

**Prerequisites**
PyCharm
pandas
scikit-learn
matplotlib
Installation

**Install the required packages:**

pip install pandas scikit-learn matplotlib

**Usage**
Place the us_daily.csv dataset in the data directory:

C:\Users\Public\PycharmProjects\PythonProject\data\us_daily.csv

**Import Libraries**

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


**Explanation Load Data:** 

Data Loading: The data will be loaded from a predefined file path using pandas. 
Feature Selection: Select features that can be used for the clustering: positiveIncrease, negativeIncrease, deathIncrease, and hospitalizedIncrease. 
Handling Missing Values: For the selected features, fill the missing values with the mean of respective columns. 
K-Means Clustering: Use the K-Means algorithm to cluster the data into 3 clusters. 
Add Cluster Labels: Cluster labels are added to the original dataset about which cluster each data point belongs to. 
Visualization: A scatter plot is done for the clusters, and the colors depict different clusters.

**Unit Testing**
There are unit tests in the application that ensure all is working as it should. Below are the different kinds of tests done:

Testing that the dataset loads up properly.
Testing for missing values and handling the data accordingly.
Test that K-Means clustering works for 3 clusters.
Test cluster labels will be added to the dataset.
