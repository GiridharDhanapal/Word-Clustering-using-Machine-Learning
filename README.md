# Word-Clustering-using-Machine-Learning
Aim of the project is to implement various machine learning clustering algorithms to cluster words based on their word embeddings. In this project, the goal is to cluster words provided in a dataset using different clustering algorithms implemented in Python.

# Unsupervised Machine Learning Algorithms Used

The **K-Means algorithm** partitions data into k clusters, where each data point belongs to the cluster with the nearest mean. The algorithm involves randomly initializing k cluster centroids and then assigning each data point to the nearest centroid. Updating the centroids based on the mean of the assigned points. Repeating the same and update steps until convergence.

**K-Means++** improves the initialization step of K-Means by selecting the first centroid randomly from the data points. Selecting subsequent centroids with a probability proportional to the squared distance from the nearest existing centroid. Then proceeding with the standard K-Means algorithm.

**Bisecting K-Means** is a hierarchical clustering algorithm that it starts with all data points in a single cluster. Iteratively selects a cluster to split using the standard K-Means algorithm. Repeats the process until the desired number of clusters is reached.

# Implementation Details

Languages: Python

Libraries Used: NumPy, SciPy, Matplotlib

Data: Words with associated 300-dimensional word embeddings.

# Conclusion

This repository contains implementations of K-Means, K-Means++, and Bisecting K-Means clustering algorithms applied to word embeddings. It provides a comprehensive analysis of clustering quality using the Silhouette coefficient and offers insights into choosing the optimal clustering approach for the given word dataset.
