
import pandas as pd
import numpy as np
from datetime import datetime

def calculate_rfm(df: pd.DataFrame, current_date: datetime = None) -> pd.DataFrame:
    """
    Calculates Recency, Frequency, and Monetary (RFM) values for each customer.

    Args:
        df (pd.DataFrame): Cleaned transactions DataFrame with 'CustomerID',
                           'InvoiceDate', and 'TotalPrice' columns.
        current_date (datetime, optional): The reference date to calculate Recency.
                                           If None, the latest InvoiceDate in the data + 1 day is used.

    Returns:
        pd.DataFrame: A DataFrame with CustomerID and their RFM scores.
    """
    if current_date is None:
        # Use the day after the last transaction date in the dataset as the reference point
        current_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        print(f"Using {current_date.strftime('%Y-%m-%d')} as the reference date for RFM calculation.")

    rfm_df = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda date: (current_date - date.max()).days),
        Frequency=('InvoiceNo', 'nunique'), # Number of unique invoices
        Monetary=('TotalPrice', 'sum')
    ).reset_index()

    print(f"RFM calculation complete. RFM DataFrame shape: {rfm_df.shape}")
    return rfm_df

def create_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates additional derived features from the transactions data.
    (Example: Average Order Value, Quantity per Order)

    Args:
        df (pd.DataFrame): Cleaned transactions DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with CustomerID and additional features.
    """
    # Calculate Average Order Value
    avg_order_value_df = df.groupby('CustomerID').agg(
        AvgOrderValue=('TotalPrice', 'mean'),
        AvgQuantityPerOrder=('Quantity', 'mean') # Example of another feature
    ).reset_index()

    print(f"Additional features created. DataFrame shape: {avg_order_value_df.shape}")
    return avg_order_value_df


# For demonstration purposes
if __name__ == "__main__":
    print("--- Testing feature_engineering.py ---")
    # Create a dummy cleaned transactions DataFrame
    from datetime import datetime, timedelta
    today = datetime.now()

    dummy_transactions = pd.DataFrame({
        'CustomerID': [1, 1, 2, 3, 1, 2, 4, 3],
        'InvoiceNo': ['A1', 'A2', 'B1', 'C1', 'A3', 'B2', 'D1', 'C2'],
        'InvoiceDate': [
            today - timedelta(days=10),
            today - timedelta(days=5),
            today - timedelta(days=20),
            today - timedelta(days=15),
            today - timedelta(days=2),
            today - timedelta(days=8),
            today - timedelta(days=1),
            today - timedelta(days=10)
        ],
        'TotalPrice': [50, 30, 100, 20, 60, 40, 70, 25],
        'Quantity': [2, 1, 5, 1, 3, 2, 4, 1]
    })
    print("Dummy Cleaned Transactions Data Head:")
    print(dummy_transactions.head())

    # Test RFM calculation
    rfm_results = calculate_rfm(dummy_transactions)
    print("\nRFM Results:")
    print(rfm_results)

    # Test additional features calculation
    additional_features = create_additional_features(dummy_transactions)
    print("\nAdditional Features Results:")
    print(additional_features)

    # Merge RFM and additional features
    combined_features = pd.merge(rfm_results, additional_features, on='CustomerID', how='left')
    print("\nCombined Features (RFM + Additional):")
    print(combined_features)

    print("\n--- feature_engineering.py testing complete ---")
