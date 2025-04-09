# import necessary libraries and packages

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# read the dataset into a dataframe
dataset = pd.read_csv('Mall_Customers.csv')

# assign x
x = dataset[['Annual Income (k$)', 'Spending Score (1-100)']]

# scale x
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

# use elbow to find optimal number of clusters

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# 5 was optimal number according to plot

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(x)

#visualize the clusters
x = pd.DataFrame(x, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
plt.figure(figsize=(10, 8))
plt.scatter(x['Annual Income (k$)'], x['Spending Score (1-100)'], c=dataset['cluster'], cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker ='X')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
