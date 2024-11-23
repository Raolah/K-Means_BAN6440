import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Data Loading: Loaded the dataset using pandas
data = pd.read_csv(r'C:\Users\Public\PycharmProjects\PythonProject\data\us_daily.csv')

# Feature Selection: Selected relevant features for clustering
features = data[['positiveIncrease', 'negativeIncrease', 'deathIncrease', 'hospitalizedIncrease']]

# Handling Missing Values: Filled missing values with the mean of the respective columns
features = features.fillna(features.mean())

# K-Means Clustering: Implemented the K-Means algorithm to cluster the data into 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(features)

# Adding Cluster Labels: Added the cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

# Visualization: Plotted the clusters using matplotlib
plt.scatter(data['positiveIncrease'], data['negativeIncrease'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Positive Increase')
plt.ylabel('Negative Increase')
plt.title('K-Means Clustering of COVID-19 Data')
plt.show()