import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class SegmentationEngine:
    """
    Customer Segmentation Module
    """

    def __init__(self):
        pass

    def kmeans_clustering(self, df):
        """
        Perform K-Means clustering.
        """
        raise NotImplementedError

    def dbscan_clustering(self, df):
        """
        Perform DBSCAN clustering.
        """
        raise NotImplementedError

    def agglomerative_clustering(self, df):
        """
        Perform Agglomerative clustering.
        """
        raise NotImplementedError

    def apply_pca(self, X):
        """
        Reduce dimensions using PCA.
        """
        raise NotImplementedError

    def apply_tsne(self, X):
        """
        Reduce dimensions using t-SNE.
        """
        raise NotImplementedError

    def plot_clusters(self, X, labels, title):
        """
        Plot clustered data.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels)
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()