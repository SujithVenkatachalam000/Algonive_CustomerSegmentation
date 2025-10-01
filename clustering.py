
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # Even if preprocessed, this acts as a placeholder or final check
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def find_optimal_clusters_kmeans(data: pd.DataFrame, max_k: int = 10) -> (int, list, list):
    """
    Finds the optimal number of clusters using the Elbow Method (SSE)
    and Silhouette Score for K-Means.

    Args:
        data (pd.DataFrame): Preprocessed DataFrame for clustering.
        max_k (int): Maximum number of clusters to check.

    Returns:
        tuple: (optimal_k_silhouette, sse_values, silhouette_scores)
               - Recommended K based on silhouette score.
               - List of SSE values for each K.
               - List of Silhouette Scores for each K.
    """
    sse = []
    silhouette_scores = []
    k_range = range(2, max_k + 1) # K-Means needs at least 2 clusters

    print(f"Finding optimal K for K-Means (checking up to {max_k} clusters)...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        sse.append(kmeans.inertia_) # Sum of squared distances of samples to their closest cluster center

        # Calculate silhouette score if there's more than one cluster and enough samples
        if k > 1 and len(data) >= k:
            score = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(np.nan) # Append NaN if score cannot be calculated

    # Find k with the highest silhouette score (excluding NaN)
    optimal_k_silhouette = 2
    if silhouette_scores and not pd.Series(silhouette_scores).isna().all():
        valid_scores = [(k_range[i], score) for i, score in enumerate(silhouette_scores) if not np.isnan(score)]
        if valid_scores:
            optimal_k_silhouette = max(valid_scores, key=lambda item: item[1])[0]
    else:
        print("Warning: Could not calculate valid silhouette scores. Defaulting optimal_k_silhouette to 2.")


    return optimal_k_silhouette, sse, silhouette_scores

def perform_kmeans_clustering(data: pd.DataFrame, n_clusters: int, random_state: int = 42) -> KMeans:
    """
    Performs K-Means clustering on the given data.

    Args:
        data (pd.DataFrame): Preprocessed DataFrame for clustering.
        n_clusters (int): The number of clusters to form.
        random_state (int): Seed for random number generation.

    Returns:
        KMeans: Fitted K-Means model.
    """
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans_model.fit(data)
    print("K-Means clustering complete.")
    return kmeans_model

def assign_clusters_to_customers(original_customer_ids: pd.Series, cluster_labels: np.ndarray) -> pd.DataFrame:
    """
    Assigns cluster labels back to the original CustomerIDs.

    Args:
        original_customer_ids (pd.Series): A Series of original CustomerIDs.
        cluster_labels (np.ndarray): An array of cluster labels.

    Returns:
        pd.DataFrame: DataFrame with CustomerID and their assigned 'Cluster'.
    """
    customer_clusters_df = pd.DataFrame({
        'CustomerID': original_customer_ids.values,
        'Cluster': cluster_labels
    })
    print(f"Clusters assigned to {len(customer_clusters_df)} customers.")
    return customer_clusters_df

# For demonstration purposes
if __name__ == "__main__":
    print("--- Testing clustering.py ---")

    # Create dummy preprocessed data (e.g., scaled RFM features)
    dummy_preprocessed_data = pd.DataFrame({
        'Recency_scaled': np.random.rand(100),
        'Frequency_scaled': np.random.rand(100),
        'Monetary_scaled': np.random.rand(100),
        'CustomerID': range(1, 101) # Assume CustomerID was kept track of
    })

    # Separate features from CustomerID for clustering
    features_for_clustering = dummy_preprocessed_data[['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']]
    customer_ids = dummy_preprocessed_data['CustomerID']

    # Test optimal K finding
    optimal_k, sse_values, silhouette_scores = find_optimal_clusters_kmeans(features_for_clustering, max_k=7)
    print(f"\nRecommended optimal K (Silhouette): {optimal_k}")

    # Plot Elbow Method and Silhouette Score
    k_range_plot = range(2, 7 + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(k_range_plot, sse_values, marker='o')
    axes[0].set_title('Elbow Method for Optimal K')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('SSE')
    axes[0].grid(True)

    axes[1].plot(k_range_plot, silhouette_scores, marker='o')
    axes[1].set_title('Silhouette Score for Optimal K')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()

    # Perform clustering with the optimal K (or a chosen K, e.g., 3)
    chosen_k = 3 # For demonstration, let's pick 3
    kmeans_model = perform_kmeans_clustering(features_for_clustering, n_clusters=chosen_k)
    print(f"\nK-Means cluster centers:\n{kmeans_model.cluster_centers_}")

    # Assign clusters back to customers
    customer_clusters = assign_clusters_to_customers(customer_ids, kmeans_model.labels_)
    print("\nCustomer Clusters Head:\n", customer_clusters.head())
    print("\nCluster Distribution:\n", customer_clusters['Cluster'].value_counts())

    print("\n--- clustering.py testing complete ---")
