import pandas as pd
import os
import sys

# Add the src directory to the system path to import modules
# This assumes you are running the notebook from the project root or 'notebooks' directory
# Adjust path if needed
src_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)
# If running from project root
# sys.path.append(os.path.abspath('./src'))

def load_csv_data(file_name: str, raw_data_path: str = 'data/raw') -> pd.DataFrame:
    """
    Loads a CSV file from the specified raw data path.

    Args:
        file_name (str): The name of the CSV file (e.g., 'customer_transactions.csv').
        raw_data_path (str): The base path to the raw data directory.
                             Defaults to 'data/raw'.

    Returns:
        pd.DataFrame: The loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        Exception: For other potential issues during file loading.
    """
    file_path = os.path.join(raw_data_path, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
    if not os.path.isfile(file_path):
        raise ValueError(f"Error: '{file_path}' is not a valid file.")

    print(f"Attempting to load data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded '{file_name}'. Shape: {df.shape}")
        return df
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Error: The file '{file_name}' is empty.")
    except Exception as e:
        raise Exception(f"An error occurred while loading '{file_name}': {e}")

def load_excel_data(file_name: str, sheet_name: str = None, raw_data_path: str = 'data/raw') -> pd.DataFrame:
    """
    Loads an Excel file from the specified raw data path.

    Args:
        file_name (str): The name of the Excel file (e.g., 'customer_data.xlsx').
        sheet_name (str, optional): The name of the sheet to load. If None,
                                    the first sheet is loaded. Defaults to None.
        raw_data_path (str): The base path to the raw data directory.
                             Defaults to 'data/raw'.

    Returns:
        pd.DataFrame: The loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        pd.errors.EmptyDataError: If the Excel file is empty.
        Exception: For other potential issues during file loading.
    """
    file_path = os.path.join(raw_data_path, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
    if not os.path.isfile(file_path):
        raise ValueError(f"Error: '{file_path}' is not a valid file.")

    print(f"Attempting to load data from: {file_path}")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Successfully loaded '{file_name}'. Shape: {df.shape}")
        return df
    except Exception as e:
        raise Exception(f"An error occurred while loading '{file_name}': {e}")

if __name__ == "__main__":
    # --- Example Usage (for testing purposes) ---
    print("--- Testing data_loader.py ---")

    # Create a dummy raw data directory if it doesn't exist
    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')
        print("Created 'data/raw' directory for testing.")

    # Create a dummy CSV file for testing
    dummy_csv_content = "CustomerID,PurchaseDate,Amount,Product\n1,2023-01-01,10.50,A\n2,2023-01-05,20.00,B\n1,2023-01-10,5.75,C"
    with open('data/raw/dummy_transactions.csv', 'w') as f:
        f.write(dummy_csv_content)
    print("Created 'data/raw/dummy_transactions.csv'")

    try:
        # Test loading a valid CSV
        df_transactions = load_csv_data('dummy_transactions.csv')
        print("\nDummy Transactions Data Head:\n", df_transactions.head())
    except Exception as e:
        print(e)

    try:
        # Test loading a non-existent file
        df_nonexistent = load_csv_data('non_existent.csv')
    except Exception as e:
        print(e)

    try:
        # Test loading an empty file (optional, requires creating an empty file)
        # with open('data/raw/empty.csv', 'w') as f:
        #     pass
        # print("Created 'data/raw/empty.csv'")
        # df_empty = load_csv_data('empty.csv')
        # print("\nEmpty CSV Data Head:\n", df_empty.head())
        pass # Commented out to avoid creating an empty file during normal run
    except Exception as e:
        print(e)

    # Clean up dummy files and directory
    if os.path.exists('data/raw/dummy_transactions.csv'):
        os.remove('data/raw/dummy_transactions.csv')
        print("Removed 'data/raw/dummy_transactions.csv'")
    # if os.path.exists('data/raw/empty.csv'):
    #     os.remove('data/raw/empty.csv')
    #     print("Removed 'data/raw/empty.csv'")
    # if os.path.exists('data/raw'):
    #     os.rmdir('data/raw') # rmdir only works on empty directories
    #     print("Removed 'data/raw' directory.")
    print("\n--- data_loader.py testing complete ---")


# Load your customer transactions data
try:
    transactions_df = load_csv_data('your_transactions_file.csv')
    print("Transactions DataFrame loaded successfully:")
    print(transactions_df.head())
except Exception as e:
    print(e)

# If you have another dataset, e.g., demographics in Excel
# try:
#     demographics_df = load_excel_data('customer_demographics.xlsx', sheet_name='Customers')
#     print("\nDemographics DataFrame loaded successfully:")
#     print(demographics_df.head())
# except Exception as e:
#     print(e)
