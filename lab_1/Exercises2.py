import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# (a)  Plot the observations.
x1 = np.array([1, 1, 0, 5, 6, 4])
x2 = np.array([4, 3, 4, 1, 2, 0])
plt.scatter(x1, x2, c='red')
plt.grid()
plt.show()

# (b)  Randomly assign a cluster label to each observation.
# You can use the choice() command from the random module in Python or choices() for Python versions 3.6 and up.
# Report the cluster labels for each observation.
np.random.seed(127)  # random seed for the randint()
cluster_labels = np.random.randint(2, size=6)
color = ['red' if elem == 0 else 'green' for elem in cluster_labels]
fig = plt.figure(figsize=(15, 8))
plt.scatter(x1, x2, c=color)
plt.grid()
plt.show()

# (c)  Compute the centroid for each cluster.
# For each cluster, add the values of all members
# Divide the total by the number of members of the cluster.
centroid_0_x1 = 0
centroid_0_x2 = 0
centroid_1_x1 = 0
centroid_1_x2 = 0
count_0 = 0
count_1 = 0

for idx, cluster in enumerate(cluster_labels):
    if cluster == 0:
        centroid_0_x1 += x1[idx]
        centroid_0_x2 += x2[idx]
        count_0 += 1
    else:
        centroid_1_x1 += x1[idx]
        centroid_1_x2 += x2[idx]
        count_1 += 1

centroid_0_x1 = centroid_0_x1 / count_0
centroid_0_x2 = centroid_0_x2 / count_0
centroid_1_x1 = centroid_1_x1 / count_1
centroid_1_x2 = centroid_1_x2 / count_1
print("Centriod for Clutser 0 is: " + str(centroid_0_x1) + ", " + str(centroid_0_x2))
print("Centriod for Clutser 1 is: " + str(centroid_1_x1) + ", " + str(centroid_1_x2))

# (d) Assign each observation to the centroid to which it is closest,
# in terms of Euclidean distance. Report the cluster labels for each observation.
# For that, plot the observations and the centroids.
for idx, cluster in enumerate(cluster_labels):
    dist_0 = (x1[idx] - centroid_0_x1) ** 2 + (x2[idx] - centroid_0_x2) ** 2
    dist_1 = (x1[idx] - centroid_1_x1) ** 2 + (x2[idx] - centroid_1_x2) ** 2
    if dist_0 > dist_1:
        cluster_labels[idx] = 0
    else:
        cluster_labels[idx] = 1

color = ['red' if elem == 0 else 'green' for elem in cluster_labels]
fig = plt.figure(figsize=(15, 8))
plt.scatter(x1, x2, c=color)
plt.grid()
plt.show()


# (f) In your plot from (a), color the observations according to the cluster labels obtained.
M = np.column_stack((x1,x2))
kmeans = KMeans(n_clusters=2, random_state=0).fit(M)
cluster_labels = kmeans.labels_

color = ['red' if elem == 0 else 'green' for elem in cluster_labels]
fig = plt.figure(figsize=(15,8))
plt.scatter(x1, x2, c=color)
plt.grid()
plt.show()


