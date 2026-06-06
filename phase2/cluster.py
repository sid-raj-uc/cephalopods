"""
Clustering module for phase2.

Loads all DINOv2 feature vectors, reduces dimensionality with PCA,
clusters with KMeans (or HDBSCAN), and saves results.

Usage:
    from phase2.cluster import load_and_prepare, cluster_kmeans, save_clusters, get_representatives
"""

import logging
from pathlib import Path

import numpy as np

CLUSTERS_DIR = Path("data/phase2/clusters")
log = logging.getLogger(__name__)


def load_and_prepare(pca_components: int = 50):
    """
    Load all feature vectors, L2-normalize, reduce with PCA.
    Returns (features_pca, features_raw, video_ids, start_secs, end_secs).
    """
    from phase2.extractor import load_all_features
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    features, video_ids, start_secs, end_secs = load_all_features()
    log.info("Loaded %d feature vectors  (dim=%d)", len(features), features.shape[1])

    features = normalize(features, norm="l2")

    n_components = min(pca_components, len(features), features.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features)
    explained = pca.explained_variance_ratio_.sum() * 100
    log.info("PCA: %d → %d dims  (%.1f%% variance retained)", features.shape[1], n_components, explained)

    return features_pca, features, video_ids, start_secs, end_secs


def cluster_kmeans(features: np.ndarray, k: int, random_state: int = 42):
    """
    Run KMeans. Returns (labels, inertia).
    k should be close to the number of behaviors in your ethogram.
    """
    from sklearn.cluster import KMeans

    log.info("Running KMeans with k=%d ...", k)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(features)

    for i in range(k):
        count = (labels == i).sum()
        log.info("  Cluster %2d : %d segments", i, count)

    log.info("Inertia: %.2f", km.inertia_)
    return labels, km


def cluster_hdbscan(features: np.ndarray, min_cluster_size: int = 3):
    """
    Run HDBSCAN. Returns labels (-1 = noise/outlier).
    Good when you don't know how many clusters to expect.
    """
    import hdbscan

    log.info("Running HDBSCAN (min_cluster_size=%d) ...", min_cluster_size)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(features)

    unique = sorted(set(labels))
    for i in unique:
        count = (labels == i).sum()
        label_str = "noise" if i == -1 else f"Cluster {i:2d}"
        log.info("  %s : %d segments", label_str, count)

    n_noise = (labels == -1).sum()
    log.info("Total clusters: %d  |  noise points: %d", len(unique) - (1 if -1 in unique else 0), n_noise)
    return labels, clusterer


def get_representatives(
    features: np.ndarray,
    labels: np.ndarray,
    n: int = 5,
) -> dict[int, list[int]]:
    """
    For each cluster, return indices of the n segments closest to the centroid.
    These are the ones to manually review and label.
    """
    reps = {}
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        idxs = np.where(mask)[0]
        centroid = features[mask].mean(axis=0)
        dists = np.linalg.norm(features[idxs] - centroid, axis=1)
        closest = idxs[np.argsort(dists)[:n]]
        reps[cluster_id] = closest.tolist()
    return reps


def pca_2d(features: np.ndarray) -> np.ndarray:
    """Reduce to 2D with PCA for scatter plot visualization."""
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(features)


def save_clusters(
    labels: np.ndarray,
    video_ids: list[str],
    start_secs: np.ndarray,
    end_secs: np.ndarray,
    name: str = "clusters",
):
    """Save cluster assignments alongside segment metadata."""
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    out = CLUSTERS_DIR / f"{name}.npz"
    np.savez_compressed(
        out,
        labels=labels,
        video_ids=np.array(video_ids),
        start_secs=start_secs,
        end_secs=end_secs,
    )
    log.info("Saved cluster assignments → %s", out)
    return out


def load_clusters(name: str = "clusters"):
    data = np.load(CLUSTERS_DIR / f"{name}.npz", allow_pickle=True)
    return (
        data["labels"],
        data["video_ids"].tolist(),
        data["start_secs"],
        data["end_secs"],
    )
