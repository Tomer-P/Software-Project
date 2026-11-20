"""Compare SymNMF and K-means clustering on the same input points."""

import sys
import numpy as np
import symnmfmodule
import kmeans
from symnmf import (
    initialize_h, read_points, validate_arguments,
    print_error, K_ERROR, OTHER_ERROR
)
from sklearn.metrics import silhouette_score, adjusted_rand_score

def get_symnmf_labels(H):
    """Assigns each point to the column with its highest H association score."""
    return np.argmax(H, axis=1)

def get_kmeans_labels(points, centroids):
    """Returns each point's nearest centroid index as its cluster label."""
    labels = []

    for point in points:
        closest = kmeans.closest_centroid(point, centroids)
        labels.append(centroids.index(closest))

    return labels

def run_symnmf(points, k):
    """Builds W, initializes H, and returns the factor matrix optimized in C."""
    points_list = points.tolist()
    n = len(points)

    W = symnmfmodule.norm(points_list)
    H = initialize_h(W, n, k)

    return symnmfmodule.symnmf(
        H.tolist(),
        W,
        n,
        k
    )

def parse_analysis_input():
    """Validate the command-line inputs and return k with the data points."""
    if len(sys.argv) != 3:
        print_error(OTHER_ERROR)
    try:
        k_value = float(sys.argv[1])
    except ValueError:
        print_error(K_ERROR)
    points = read_points(sys.argv[2])
    validate_arguments(k_value, "symnmf", len(points))
    return int(k_value), points


def main():
    """Read k and an input file, then print two silhouette scores and ARI."""
    k, points = parse_analysis_input()

    H = run_symnmf(points, k)
    nmf_labels = get_symnmf_labels(H)

    centroids = kmeans.k_means(
        points.tolist(),
        k,
        300
    )

    kmeans_labels = get_kmeans_labels(
        points.tolist(),
        centroids
    )

    try:
        nmf_score = silhouette_score(points, nmf_labels)
        kmeans_score = silhouette_score(points, kmeans_labels)
    except ValueError:
        # A silhouette score needs at least two distinct clusters.
        print("An Error Has Occurred")
        sys.exit(1)
    ari = adjusted_rand_score(nmf_labels, kmeans_labels)

    print("nmf: %.4f" % nmf_score)
    print("kmeans: %.4f" % kmeans_score)
    print("ari: %.4f" % ari)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print_error(OTHER_ERROR)
