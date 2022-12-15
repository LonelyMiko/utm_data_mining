import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

data = np.array([
    [0, 0.3, 0.4, 0.7],
    [0.3, 0, 0.5, 0.8],
    [0.4, 0.5, 0, 0.45],
    [0.7, 0.8, 0.45, 0]
])
labels_for_data = ["1", "2", "3", "4"]

# (a)  On the basis of this dissimilarity matrix,
# sketch the dendrogram that results from hierarchically clustering these four observations using complete linkage.
# Be sure to indicate on the plot theheight at which each fusion occurs,
# as well as the observations corresponding to each leaf in thedendrogram.
linkage_data = linkage(data, method='complete', metric='chebyshev')
dendrogram(linkage_data)
plt.show()
complete_cut_tree = cut_tree(linkage_data, n_clusters=2).T

# (b)  Repeat (a), this time using single linkage clustering.
linkage_data = linkage(data, method='single', metric='chebyshev')
dendrogram(linkage_data, labels=labels_for_data)
single_cut_tree = cut_tree(linkage_data, n_clusters=2).T


plt.show()
# c  Suppose that we cut the dendogram obtained in a) such tha two clusters result. Which observations are in each
# cluster?
print(complete_cut_tree) # Result is: {1,2} is in one cluster, {3,4} in another

# Cut the dendrogram obtained in (b) such that two clusters result.
# Which observations are ineach cluster ? Compare the results with those obtained in the previous question
print(single_cut_tree) # Result is: {1,2,3} is in one cluster, {4} in another




