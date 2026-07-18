"""
customer_analytics/segmentation_engines.py
--------------------------------------------
Member 4 - Sub-Module D: Strategic Demographic Partitions (Clustering)

What this file does (plain English):
1. Takes the customer_features table built by
   core_pipeline/transformation_pipes.py (RFM-style: spend, frequency,
   recency-ish gap, etc).
2. Scales it and runs 3 clustering algorithms: KMeans, DBSCAN, Agglomerative.
3. Uses the Elbow Method + Silhouette Score to justify the choice of k
   for KMeans.
4. Reduces dimensions with PCA, t-SNE, and UMAP purely for visualization
   (2D scatter plots colored by cluster).
5. Saves all charts to analytical_reports/ so they can go straight into
   the slides + dashboard.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[WARN] umap-learn not installed. Run: pip install umap-learn")


FEATURE_COLS = [
    "total_spend", "avg_order_value", "num_orders",
    "avg_gap_days", "weekend_order_ratio", "country_target_enc",
]


# ----------------------------------------------------------------------
# 1. LOAD + SCALE
# ----------------------------------------------------------------------
def load_features(path="analytical_reports/customer_features.csv"):
    return pd.read_csv(path)


def scale_features(df, feature_cols=FEATURE_COLS):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].fillna(0))
    return X_scaled, scaler


# ----------------------------------------------------------------------
# 2. ELBOW METHOD + SILHOUETTE SCORE  -- "how many clusters (k) is right?"
# ----------------------------------------------------------------------
def elbow_and_silhouette(X_scaled, k_range=range(2, 10), save_path="analytical_reports"):
    """
    Elbow method: plot inertia (within-cluster sum of squares) vs k.
    The 'elbow' point (where the curve bends) is a good candidate for k.

    Silhouette score: measures how well-separated clusters are
    (ranges -1 to 1, higher is better). We pick the k with the best score.
    """
    os.makedirs(save_path, exist_ok=True)
    inertias = []
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("k (number of clusters)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), sil_scores, marker="o", color="orange")
    axes[1].set_title("Silhouette Scores")
    axes[1].set_xlabel("k (number of clusters)")
    axes[1].set_ylabel("Silhouette Score")

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "elbow_silhouette.png"))
    plt.close(fig)

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"Best k by silhouette score: {best_k}")
    return best_k, inertias, sil_scores


# ----------------------------------------------------------------------
# 3. THE THREE CLUSTERING ALGORITHMS
# ----------------------------------------------------------------------
def run_kmeans(X_scaled, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    return labels, model


def run_dbscan(X_scaled, eps=0.8, min_samples=5):
    """
    DBSCAN finds clusters based on density and automatically flags
    outliers as noise (label = -1). No need to specify k in advance.
    eps/min_samples usually need tuning per dataset.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    return labels, model


def run_agglomerative(X_scaled, n_clusters):
    """
    Agglomerative (hierarchical) clustering: starts with every point as
    its own cluster and merges the closest pairs step by step until
    n_clusters remain.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)
    return labels, model


# ----------------------------------------------------------------------
# 4. DIMENSIONALITY REDUCTION FOR VISUALIZATION
# ----------------------------------------------------------------------
def reduce_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X_scaled), pca


def reduce_tsne(X_scaled, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30, init="pca")
    return tsne.fit_transform(X_scaled)


def reduce_umap(X_scaled, n_components=2):
    if not UMAP_AVAILABLE:
        return None
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(X_scaled)


def plot_clusters_2d(coords, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=20)
    ax.set_title(title)
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# 5. MAIN — run all three algorithms + all three reduction techniques
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = load_features()
    X_scaled, scaler = scale_features(df)

    best_k, inertias, sil_scores = elbow_and_silhouette(X_scaled)

    # --- run all 3 clustering algorithms ---
    kmeans_labels, kmeans_model = run_kmeans(X_scaled, n_clusters=best_k)
    dbscan_labels, dbscan_model = run_dbscan(X_scaled)
    agglo_labels, agglo_model = run_agglomerative(X_scaled, n_clusters=best_k)

    df["cluster_kmeans"] = kmeans_labels
    df["cluster_dbscan"] = dbscan_labels
    df["cluster_agglomerative"] = agglo_labels

    print("KMeans silhouette:", silhouette_score(X_scaled, kmeans_labels))
    if len(set(dbscan_labels)) > 1:
        print("DBSCAN silhouette:", silhouette_score(X_scaled, dbscan_labels))
    print("Agglomerative silhouette:", silhouette_score(X_scaled, agglo_labels))

    # --- dimensionality reduction for plotting ---
    os.makedirs("analytical_reports", exist_ok=True)

    pca_coords, pca_model = reduce_pca(X_scaled)
    plot_clusters_2d(pca_coords, kmeans_labels, "KMeans Clusters (PCA projection)",
                      "analytical_reports/clusters_pca.png")

    tsne_coords = reduce_tsne(X_scaled)
    plot_clusters_2d(tsne_coords, kmeans_labels, "KMeans Clusters (t-SNE projection)",
                      "analytical_reports/clusters_tsne.png")

    if UMAP_AVAILABLE:
        umap_coords = reduce_umap(X_scaled)
        plot_clusters_2d(umap_coords, kmeans_labels, "KMeans Clusters (UMAP projection)",
                          "analytical_reports/clusters_umap.png")

    df.to_csv("analytical_reports/customer_segments.csv", index=False)
    print(f"Saved segmented customer table with {df.shape[0]} rows.")