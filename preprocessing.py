
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline # Import Pipeline
import numpy as np

def clean_transactions_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial cleaning on the raw transactions DataFrame.
    Assumes columns like 'CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice'.

    Args:
        df (pd.DataFrame): Raw transactions DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # 1. Handle missing CustomerID (critical for segmentation)
    # Remove rows where CustomerID is missing as we cannot segment them
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int) # Convert to integer type

    # 2. Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # 3. Remove transactions with negative Quantity (returns)
    df = df[df['Quantity'] > 0]

    # 4. Remove transactions with negative UnitPrice (errors)
    df = df[df['UnitPrice'] > 0]

    # 5. Calculate TotalPrice for each transaction item
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # 6. Remove duplicate rows (if a transaction item is recorded identically twice)
    df.drop_duplicates(inplace=True)

    print(f"Initial cleaning complete. DataFrame shape: {df.shape}")
    return df

def preprocess_for_clustering(df: pd.DataFrame, numerical_features: list, categorical_features: list = None) -> (pd.DataFrame, ColumnTransformer):
    """
    Applies scaling to numerical features and one-hot encoding to categorical features.
    Returns the preprocessed DataFrame and the fitted preprocessor.

    Args:
        df (pd.DataFrame): DataFrame with features for clustering.
        numerical_features (list): List of column names to scale.
        categorical_features (list, optional): List of column names to one-hot encode.
                                               Defaults to None.

    Returns:
        tuple: (pd.DataFrame, ColumnTransformer)
               - Preprocessed DataFrame.
               - Fitted ColumnTransformer object.
    """
    preprocessor_steps = []

    # Numerical feature processing: Impute then Scale
    if numerical_features:
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        preprocessor_steps.append(('num', numerical_transformer, numerical_features))

    # Categorical feature processing: Impute then One-Hot Encode
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # Impute before encoding
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor_steps.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=preprocessor_steps,
        remainder='passthrough' # Keep other columns (like CustomerID if not in features)
    )

    # Fit and transform
    transformed_data = preprocessor.fit_transform(df)

    # Get feature names after transformation
    new_numerical_features = numerical_features
    if categorical_features:
        # Get one-hot encoded feature names
        onehot_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        all_features = numerical_features + list(onehot_features)
    else:
        all_features = numerical_features

    processed_df = pd.DataFrame(transformed_data, columns=all_features, index=df.index)

    print(f"Preprocessing for clustering complete. Processed DataFrame shape: {processed_df.shape}")
    return processed_df, preprocessor

# For demonstration purposes, if you want to run this script directly
if __name__ == "__main__":
    print("--- Testing preprocessing.py ---")
    # Create a dummy DataFrame similar to raw transactions
    data = {
        'InvoiceNo': ['536365', '536365', '536366', '536367', '536367', '536368', '536369', '536369', '536370', '536371'],
        'StockCode': ['85123A', '71053', '22752', '22752', '84879', '22752', '22752', '84879', '84879', '71053'],
        'Description': ['WHITE HANGING HEART T-LIGHT HOLDER', 'WHITE METAL LANTERN', 'SET 7 BABUSHKA NESTING BOXES', 'SET 7 BABUSHKA NESTING BOXES', 'ASSORTED COLOUR BIRD ORNAMENT', 'SET 7 BABUSHKA NESTING BOXES', 'SET 7 BABUSHKA NESTING BOXES', 'ASSORTED COLOUR BIRD ORNAMENT', 'ASSORTED COLOUR BIRD ORNAMENT', 'WHITE METAL LANTERN'],
        'Quantity': [6, 6, 2, 2, 32, 2, 4, 32, -1, 6], # Added negative quantity for test
        'InvoiceDate': ['2010-12-01 08:26:00', '2010-12-01 08:26:00', '2010-12-01 08:28:00', '2010-12-01 08:34:00', '2010-12-01 08:34:00', '2010-12-01 08:34:00', '2010-12-01 08:35:00', '2010-12-01 08:35:00', '2010-12-01 08:41:00', '2010-12-01 08:45:00'],
        'UnitPrice': [2.55, 3.39, 7.65, 7.65, 1.69, 7.65, 7.65, 1.69, -1.0, 3.39], # Added negative unit price for test
        'CustomerID': [17850, 17850, 17850, 13047, 13047, None, 13047, 13047, 17850, 17850], # Added None for test
        'Country': ['United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom']
    }
    dummy_df = pd.DataFrame(data)

    print("Original Dummy Data Head:")
    print(dummy_df.head())
    print("\nOriginal Dummy Data Info:")
    dummy_df.info()

    # Test cleaning function
    cleaned_df = clean_transactions_data(dummy_df.copy()) # Use .copy() to avoid modifying original
    print("\nCleaned Data Head:")
    print(cleaned_df.head())
    print("\nCleaned Data Info:")
    cleaned_df.info()

    # Dummy data for clustering preprocessing (e.g., after RFM calculation)
    clustering_data = pd.DataFrame({
        'CustomerID': [1, 2, 3, 4, 5],
        'Recency': [10, 50, 120, 5, 200],
        'Frequency': [5, 1, 2, 10, 1],
        'Monetary': [200, 30, 80, 500, 10],
        'PreferredCategory': ['Electronics', 'Books', 'Electronics', 'Books', 'Clothing']
    })
    print("\nOriginal Clustering Data Head:")
    print(clustering_data.head())

    numerical_cols = ['Recency', 'Frequency', 'Monetary']
    categorical_cols = ['PreferredCategory']

    preprocessed_clustering_df, preprocessor_obj = preprocess_for_clustering(
        clustering_data.drop('CustomerID', axis=1), # Drop CustomerID before scaling/encoding
        numerical_cols,
        categorical_cols
    )
    print("\nPreprocessed Clustering Data Head:")
    print(preprocessed_clustering_df.head())
    print("\nPreprocessed Clustering Data Description:")
    print(preprocessed_clustering_df.describe())
    print("\nPreprocessor Object Type:", type(preprocessor_obj))

    print("\n--- preprocessing.py testing complete ---")
