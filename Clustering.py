import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('dataset', header=None, sep='[\n ]+')
df1 = df.drop(df.columns[0], axis=1)
input_data = df1.values
np.random.seed(25)
random.seed(25)
k_values = [1,2,3,4,5,6,7,8,9]


def k_means_clustering(X, k, iters=20, seed=25):
  random.seed(seed)
  centroids = X[random.sample(range(len(X)), k)]  # initializing random centroids
  for i in range(iters):  # iterating over number of iterations
    prev_centroids = np.copy(centroids)
    Minimum_distance, centroids = update_centroids(X, centroids, k)
    if np.all(prev_centroids == centroids):
      break
  return Minimum_distance, centroids

def update_centroids(X, centroids, k):
  Euclidean_distance = []
  for i in range(X.shape[0]):  # To loop over the number of rows in our data
    distance = []
    for j in range(k):
      distance.append(sum((X[i] - centroids[j])**2))  # Calculate distance for each centroid
    Euclidean_distance.append(distance)
  Minimum_distance = np.argmin(Euclidean_distance, axis=1)  # Index of minimum distance in each row
  for j in range(k):
    centroids[j] = np.mean(X[Minimum_distance == j], axis=0)  # Update cluster centroid position
  return Minimum_distance, centroids

def k_plusplus(X, k, seed=25):
  random.seed(seed)
  centroids = [X[random.randint(0, len(X) - 1)]]  # Initialize a single centroid
  while len(centroids) < k:
    distances = []
    for i in range(len(X)):
      dist = []
      for j in range(len(centroids)):
        euclidean_distance = np.sqrt(np.sum(np.square(X[i] - centroids[j])))
        dist.append(euclidean_distance)  # Calculate distance to each centroid
      distances.append(min(dist))  # Minimum distance to any centroid
    distances = np.array(distances)
    probabilities = distances / np.sum(distances)  # Probability based on distances
    new_centroid = X[np.random.choice(len(X), p=probabilities)]
    centroids.append(new_centroid)
  return np.array(centroids)


def k_plusplus_clustering(X, k, iters=20):
  centroids = k_plusplus(X, k)  # Initialize centroids using k-means++
  for i in range(iters):  # Iterate for the specified number of iterations
    prev_centroids = np.copy(centroids)
    Minimum_distance, centroids = update_centroids(X, centroids, k)
    if np.all(prev_centroids == centroids):  # Stop if centroids don't change
      break
  return Minimum_distance, centroids


def Kmeans_centroid(X, k, max_iterations=100):
  n = X.shape[0]
  centroids = X[np.random.choice(n, k)]  # Initialize centroids randomly
  for i in range(max_iterations):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Calculate distances
    clusters = np.argmin(distances, axis=1)  # Assign clusters based on minimum distance
    new_centroids = np.array([X[clusters == j].mean(axis=0) for j in range(k)])  # Update centroids by mean
    if np.allclose(new_centroids, centroids):  # Stop if centroids don't change significantly
      break
    centroids = new_centroids
  return clusters


def bisecting_kmeans(X, k):
    clusters = [X]
    while len(clusters) < k:
        large = max(clusters, key=lambda c: len(c)) # selecting largest cluster
        cluster = Kmeans_centroid(large, 2)
        clusters.remove(large)
        clusters.append(large[cluster == 1])
        clusters.append(large[cluster == 0])
    return clusters

def silhouette(X, Minimum_distance, centroids):
  n = X.shape[0]
  k = centroids.shape[0]
  s = np.zeros(n)
  a = np.zeros(n)
  b = np.zeros(n)
  for i in range(n):
    index = Minimum_distance[i]
    # Intracuster Distance
    a[i] = np.mean(np.linalg.norm(X[i] - X[Minimum_distance == index], axis=1))
    # Average Distance to points samecluster
    # Intercluster distance
    b[i] = np.inf  # initializing the silhouette distance of i to infinity
    for j in range(k):
      if j != index:
        dist = np.mean(np.linalg.norm(X[i] - X[Minimum_distance == j], axis=1))
        b[i] = min(b[i], dist)  # updating the values of b[i] with minimal distance
    s[i] = (b[i] - a[i]) / max(a[i], b[i])  # Silhouette calculation for i
  # Return the average silhouette coefficient
  return np.mean(s)

def silhouette_coefficient_bis(X, clusters):
  n = len(X)
  a = np.zeros(n)
  b = np.zeros(n)
  for i in range(n):
    intra = np.mean(np.linalg.norm(X[i] - X[clusters == clusters[i]], axis=1))  # Intracluster distance
    inter = []
    for k in set(clusters):
      if k != clusters[i] and np.sum(clusters == k) > 0:
        new_inter = np.mean(np.linalg.norm(X[i] - X[clusters == k], axis=1))  # Intercluster distance for each cluster
        inter.append(new_inter)
    if len(inter) == 0:
      inter = [np.inf]  # Set inter to infinity if no points in other clusters
    a[i] = intra
    b[i] = np.min(inter)  # Minimum intercluster distance

  s = (b - a) / np.maximum(a, b)  # Silhouette coefficient for each point
  return np.mean(s)  # Average silhouette coefficient

# Plotting Kmeans

silhouette_scores_kmeans = []
for k in k_values:
 Minimum_distance, centroids = k_means_clustering(input_data, k)
 score_kmeans = silhouette(input_data, Minimum_distance, centroids)
 silhouette_scores_kmeans.append(score_kmeans)
 print("k =", k, "Silhouette score for kmeans:", score_kmeans)
plt.plot(k_values, silhouette_scores_kmeans,'*-')
plt.xlabel('k')
plt.ylabel('Silhouette coefficient for kmeans')
plt.title('K-means silhoutte values plotting')
plt.show()

# Plotting kmeans++

silhouette_scores_pp = []
for k in k_values:
 Minimum_distance, centroids = k_plusplus_clustering(input_data, k)
 score_pp = silhouette(input_data, Minimum_distance, centroids)
 silhouette_scores_pp.append(score_pp)
 print("k =", k, "Silhouette score for k++ :", score_pp)
plt.plot(k_values, silhouette_scores_pp,'*-')
plt.xlabel('k')
plt.ylabel('Silhouette score for k++')
plt.title('K-means++ clustering')
plt.show()

# Plotting Bisecting Kmeans

silhouette_scores = []
for k in k_values:
  clusters = bisecting_kmeans(input_data, k)
  cluster = np.zeros(len(input_data), dtype=np.int32)
  for i in range(len(input_data)):
    for j, c_j in enumerate(clusters):
      if input_data[i] in c_j:
        cluster[i] = j
        break  # Exit inner loop once cluster is found
  score = silhouette_coefficient_bis(input_data, cluster)
  silhouette_scores.append(score)
  print('k = {}, Silhouette Coefficient of Bisecting Kmeans = {}'.format(k, score))
plt.plot(k_values, silhouette_scores,'*-')
plt.title('Bisecting K-Means clustering')
plt.xlabel('k')
plt.ylabel('Silhouette scores for Bisecting')
plt.show()