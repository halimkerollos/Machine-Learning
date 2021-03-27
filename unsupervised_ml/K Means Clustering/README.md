This is the first of my unsupervised ML model demonstration notebooks 
where I apply KMeans to datasets in order to label them into whatever number of clusters makes the most sense
for the algo.
K means clustering is an unsupervised machine learning algo that attempts to group similar clusters in the data together.
In general we chose a number of clusters "k" and randomly assign each point to a cluster.
For each cluster the mean vector points are computed to produce the cluster centroid.
Each data point is then assigned to the cluster with the closest centroid. Repeating this iteration until the results are acceptable to the algo.
