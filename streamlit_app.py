
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the src directory to the system path
# This assumes the app/streamlit_app.py is run from the project root or app/ directory
# Adjust path if needed:
# If running from app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# If running from project root (and app/streamlit_app.py is in app/)
# sys.path.append(os.path.abspath('./src'))

# Import functions from src
from data_loader import load_csv_data # Assuming we'll load the processed data for the dashboard
from preprocessing import preprocess_for_clustering # Not directly used for dashboard data, but good to know
from feature_engineering import calculate_rfm # Not directly used for dashboard data
from clustering import perform_kmeans_clustering, assign_clusters_to_customers # Not directly used for dashboard data
from visualization import plot_segment_profiles, plot_correlation_matrix, plot_feature_distribution, plot_cluster_3d


# --- Configuration ---
st.set_page_config(layout="wide", page_title="Customer Segmentation Dashboard")

# --- Load Data (Simulated for Dashboard) ---
@st.cache_data # Cache data loading to prevent reloading on every interaction
def load_processed_data():
    """
    Loads the final processed data with cluster assignments.
    In a real scenario, this would load data from `data/processed/`
    """
    # This is dummy data for demonstration.
    # In a real scenario, you'd load your 'final_customer_segments.csv'
    # df = load_csv_data('final_customer_segments.csv', raw_data_path='data/processed')

    np.random.seed(42)
    num_customers = 500
    clusters = np.random.randint(0, 4, num_customers) # 4 clusters

    data = {
        'CustomerID': range(10001, 10001 + num_customers),
        'Recency': np.random.randint(1, 365, num_customers),
        'Frequency': np.random.randint(1, 50, num_customers),
        'Monetary': np.random.randint(10, 5000, num_customers),
        'AvgOrderValue': np.random.uniform(5, 200, num_customers),
        'PreferredCategory_Electronics': np.random.rand(num_customers) > 0.5,
        'PreferredCategory_Books': np.random.rand(num_customers) > 0.5,
        'Age': np.random.randint(18, 70, num_customers),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], num_customers),
        'Cluster': clusters # Assigning the generated clusters
    }
    df = pd.DataFrame(data)

    # Make RFM values somewhat reflective of clusters for better viz
    df.loc[df['Cluster'] == 0, 'Recency'] = np.random.randint(1, 30, (df['Cluster'] == 0).sum()) # Recent
    df.loc[df['Cluster'] == 0, 'Frequency'] = np.random.randint(10, 50, (df['Cluster'] == 0).sum()) # Frequent
    df.loc[df['Cluster'] == 0, 'Monetary'] = np.random.randint(1000, 5000, (df['Cluster'] == 0).sum()) # High Value

    df.loc[df['Cluster'] == 1, 'Recency'] = np.random.randint(100, 300, (df['Cluster'] == 1).sum()) # Less Recent
    df.loc[df['Cluster'] == 1, 'Frequency'] = np.random.randint(1, 5, (df['Cluster'] == 1).sum()) # Infrequent
    df.loc[df['Cluster'] == 1, 'Monetary'] = np.random.randint(10, 100, (df['Cluster'] == 1).sum()) # Low Value

    df.loc[df['Cluster'] == 2, 'Recency'] = np.random.randint(30, 150, (df['Cluster'] == 2).sum()) # Medium Recency
    df.loc[df['Cluster'] == 2, 'Frequency'] = np.random.randint(5, 20, (df['Cluster'] == 2).sum()) # Medium Frequency
    df.loc[df['Cluster'] == 2, 'Monetary'] = np.random.randint(100, 1000, (df['Cluster'] == 2).sum()) # Medium Value

    return df

# --- Dashboard Layout ---
st.title("Customer Segmentation Dashboard")

st.write("""
This dashboard visualizes customer segments based on their behavior.
""")

# Load the data
processed_df = load_processed_data()

# Display raw data (optional)
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(processed_df)

# --- Segment Analysis ---
st.subheader("Customer Segment Analysis")

# Display segment distribution
st.write("Segment Distribution:")
segment_counts = processed_df['Cluster'].value_counts().sort_index()
st.bar_chart(segment_counts)

# Display segment profiles (using the visualization function)
st.subheader("Segment Profiles")
features_to_profile = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue']
# Note: The visualization functions (like plot_segment_profiles) from visualization.py
# will need to be adapted to work within Streamlit's plotting methods (e.g., using st.pyplot).
# For simplicity in this example, we'll just call the functions, which will
# generate matplotlib figures. In a real Streamlit app, you'd capture the figure
# and pass it to st.pyplot().
# plot_segment_profiles(processed_df, 'Cluster', features_to_profile) # Pass the dataframe and features

# Example of adapting a plot function for Streamlit:
def st_plot_feature_distribution(df: pd.DataFrame, column: str, bins: int = 30, title: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=bins, ax=ax)
    ax.set_title(title if title else f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', alpha=0.75)
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free memory

def st_plot_segment_profiles(df: pd.DataFrame, segment_column: str, features: list):
    """
    Plots box plots for specified features across different customer segments using st.pyplot.
    """
    if not features:
        st.write("No features provided for plotting segment profiles.")
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
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free memory

# Use the Streamlit-adapted plotting function
st_plot_segment_profiles(processed_df, 'Cluster', features_to_profile)


# Add interactive elements (examples)
st.sidebar.header("Filter Data")
selected_cluster = st.sidebar.multiselect(
    "Select Clusters",
    options=sorted(processed_df['Cluster'].unique()),
    default=sorted(processed_df['Cluster'].unique())
)

filtered_df = processed_df[processed_df['Cluster'].isin(selected_cluster)]

st.subheader("Filtered Data Analysis")
st.write(f"Showing data for {len(filtered_df)} customers in selected clusters.")

if not filtered_df.empty:
    # Example: Distribution of a feature for filtered data
    st_plot_feature_distribution(filtered_df, 'Monetary', title='Monetary Distribution (Filtered)')

    # Example: Correlation matrix for filtered data
    numerical_cols_filtered = filtered_df.select_dtypes(include=np.number).columns.tolist()
    if 'CustomerID' in numerical_cols_filtered:
        numerical_cols_filtered.remove('CustomerID') # Exclude CustomerID from correlation

    if numerical_cols_filtered:
        st.write("Correlation Matrix (Filtered Data):")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        corr_matrix = filtered_df[numerical_cols_filtered].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
        st.pyplot(fig_corr)
        plt.close(fig_corr)
    else:
        st.write("No numerical features to plot correlation for filtered data.")


else:
    st.write("No data to display for the selected clusters.")

# --- Add other sections as needed ---
# Example: 3D Plotting (if applicable and features are suitable)
st.subheader("3D Cluster Visualization")
st.write("Visualize clusters in a 3D space using selected features (e.g., RFM).")

# Note: 3D plotting in Streamlit requires careful handling of matplotlib figures.
# You would need to adapt the plot_cluster_3d function similarly to st_plot_feature_distribution.
# Due to the complexity and potential dependencies, this is left as a placeholder.
st.info("3D plot functionality is a placeholder and requires adaptation.")

# --- Footer ---
st.markdown("---")
st.markdown("Dashboard created using Streamlit")

# To run this app:
# 1. Save this code as a Python file (e.g., app/streamlit_app.py).
# 2. Open your terminal or command prompt.
# 3. Navigate to the directory containing the 'app' folder.
# 4. Run the command: streamlit run app/streamlit_app.py
# 5. Streamlit will open the app in your web browser.
# In Colab, you might need to use ngrok or a similar service for external access.
"""
