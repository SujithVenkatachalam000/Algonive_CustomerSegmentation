
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_feature_distribution(df: pd.DataFrame, column: str, bins: int = 30, title: str = None):
    """
    Plots the distribution of a numerical feature using a histogram and KDE.

    Args:
        df (pd.DataFrame): DataFrame containing the feature.
        column (str): Name of the column to plot.
        bins (int): Number of bins for the histogram.
        title (str, optional): Custom plot title. Defaults to None.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=bins)
    plt.title(title if title else f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, title: str = "Correlation Matrix"):
    """
    Plots a heatmap of the correlation matrix for numerical features.

    Args:
        df (pd.DataFrame): DataFrame containing numerical features.
        title (str, optional): Custom plot title. Defaults to "Correlation Matrix".
    """
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(title)
    plt.show()

def plot_segment_profiles(df: pd.DataFrame, segment_column: str, features: list):
    """
    Plots box plots for specified features across different customer segments.

    Args:
        df (pd.DataFrame): DataFrame containing segments and features.
        segment_column (str): Name of the column representing customer segments.
        features (list): List of numerical feature columns to plot.
    """
    if not features:
        print("No features provided for plotting segment profiles.")
        return

    num_features = len(features)
    fig, axes = plt.subplots(num_features, 1, figsize=(10, num_features * 5))
    if num_features == 1:
        axes = [axes] # Ensure axes is iterable even for a single subplot

    for i, feature in enumerate(features):
        sns.boxplot(x=segment_column, y=feature, data=df, ax=axes[i])
        axes[i].set_title(f'{feature} by {segment_column}')
        axes[i].set_xlabel(segment_column)
        axes[i].set_ylabel(feature)
        axes[i].grid(axis='y', alpha=0.75)

    plt.tight_layout()
    plt.show()

def plot_cluster_3d(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, cluster_col: str, title: str = "3D Cluster Visualization"):
    """
    Plots a 3D scatter plot of clusters using three features.

    Args:
        df (pd.DataFrame): DataFrame with features and cluster assignments.
        x_col (str): Column name for the x-axis.
        y_col (str): Column name for the y-axis.
        z_col (str): Column name for the z-axis.
        cluster_col (str): Column name for cluster labels.
        title (str, optional): Custom plot title. Defaults to "3D Cluster Visualization".
    """
    # This requires 'mpl_toolkits.mplot3d' for 3D plotting
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(df[x_col], df[y_col], df[z_col], c=df[cluster_col], cmap='viridis', s=50, alpha=0.8)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(title)
    plt.colorbar(scatter, label=cluster_col)
    plt.show()

# For demonstration purposes
if __name__ == "__main__":
    print("--- Testing visualization.py ---")

    # Dummy data for plotting
    np.random.seed(42)
    data = pd.DataFrame({
        'FeatureA': np.random.normal(loc=10, scale=2, size=100),
        'FeatureB': np.random.normal(loc=50, scale=10, size=100),
        'FeatureC': np.random.randint(1, 5, size=100),
        'Cluster': np.random.randint(0, 3, size=100),
        'Recency': np.random.randint(1, 365, size=100),
        'Frequency': np.random.randint(1, 20, size=100),
        'Monetary': np.random.randint(10, 1000, size=100)
    })
    # Add some correlation for correlation matrix
    data['FeatureD'] = data['FeatureA'] * 2 + np.random.normal(loc=0, scale=1, size=100)

    # Test feature distribution plot
    plot_feature_distribution(data, 'FeatureA', title='Distribution of Feature A')

    # Test correlation matrix plot
    plot_correlation_matrix(data[['FeatureA', 'FeatureB', 'FeatureC', 'FeatureD']], title='Feature Correlation')

    # Test segment profiles plot (using dummy RFM and clusters)
    plot_segment_profiles(data, 'Cluster', ['Recency', 'Frequency', 'Monetary'])

    # Test 3D cluster plot (ensure you have matplotlib with 3d capabilities)
    try:
        plot_cluster_3d(data, 'Recency', 'Frequency', 'Monetary', 'Cluster', 'Customer Segments in RFM Space')
    except ImportError:
        print("\nSkipping 3D plot: mpl_toolkits.mplot3d might not be available or compatible.")
    except Exception as e:
        print(f"\nError during 3D plot: {e}")

    print("\n--- visualization.py testing complete ---")
