"""
A helper module for loading datasets for the course project. You should not change this file.
Please ensure all data files are located in a './data/' subdirectory
relative to this script.
"""

import numpy as np
import pandas as pd
import os

def load_gisette_local(data_path='./data'):
    """
    Loads the Gisette dataset (training and validation sets) from local files.
    The validation set is used as the test set.

    Args:
        data_path (str): The path to the directory containing the data files.

    Returns:
        tuple: A tuple containing (X_train, y_train, X_test, y_test).
               Returns (None, None, None, None) if files are not found.
    """
    print("Loading Gisette dataset from local files...")
    try:
        train_data_file = os.path.join(data_path, 'gisette_train.data')
        train_labels_file = os.path.join(data_path, 'gisette_train.labels')
        test_data_file = os.path.join(data_path, 'gisette_valid.data')
        test_labels_file = os.path.join(data_path, 'gisette_valid.labels')

        X_train = np.loadtxt(train_data_file).astype(np.float32)
        y_train = np.loadtxt(train_labels_file).astype(int)
        X_test = np.loadtxt(test_data_file).astype(np.float32)
        y_test = np.loadtxt(test_labels_file).astype(int)

        # Convert labels from {-1, 1} to {0, 1} for consistency
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0

        print("Gisette loaded successfully.")
        return X_train, y_train, X_test, y_test

    except FileNotFoundError:
        print("Error: Gisette data files not found in './data/' directory.")
        print("Please ensure all 4 Gisette files are downloaded.\n")
        return None, None, None, None

def load_higgs_subset_local(num_samples=500000, data_path='./data'):
    """
    Loads a subset of the HIGGS dataset from a local .csv.gz file.

    Args:
        num_samples (int): The number of samples to load from the top of the file.
        data_path (str): The path to the directory containing the data file.

    Returns:
        tuple: A tuple containing (X, y).
               Returns (None, None) if the file is not found.
    """
    print(f"\nLoading a subset of {num_samples} samples from HIGGS dataset...")
    try:
        higgs_filepath = os.path.join(data_path, 'HIGGS.csv.gz')
        df = pd.read_csv(higgs_filepath, header=None, nrows=num_samples)
        data = df.to_numpy()

        # The first column is the label, the rest are features
        X = data[:, 1:].astype(np.float32)
        y = data[:, 0].astype(int)

        print("HIGGS subset loaded successfully.")
        return X, y

    except FileNotFoundError:
        print(f"Error: HIGGS.csv.gz not found at {higgs_filepath}")
        print("Please download it first.\n")
        return None, None

if __name__ == '__main__':
    print("--- Testing data loading module ---")

    # Test the Gisette loader
    X_train_g, y_train_g, X_test_g, y_test_g = load_gisette_local()
    if X_train_g is not None:
        print(f"Gisette shapes: X_train={X_train_g.shape}, X_test={X_test_g.shape}")

    # Test the HIGGS loader (loading a small subset for a quick test)
    X_higgs, y_higgs = load_higgs_subset_local(num_samples=10000)
    if X_higgs is not None:
        print(f"HIGGS subset shapes: X={X_higgs.shape}, y={y_higgs.shape}")