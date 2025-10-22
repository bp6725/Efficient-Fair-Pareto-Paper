"""
Data Loading Utilities

Functions for loading and preprocessing FARE benchmark datasets.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def format_fare_input(fare_dict: dict) -> pd.DataFrame:
    """
    Format FARE dataset dictionary into a DataFrame.
    
    Parameters
    ----------
    fare_dict : dict
        Dictionary with keys 0 (features), 1 (sensitive), 2 (labels)
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with features, 'attr' (sensitive), and 'label' columns
    """
    X, S, Y = fare_dict[0], fare_dict[1], fare_dict[2]
    
    # Combine into single array
    combined = np.concatenate([X, S.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    
    # Create column names
    n_features = X.shape[1]
    columns = list(range(n_features)) + ['attr', 'label']
    
    return pd.DataFrame(data=combined, columns=columns)


def load_fare_dataset(
    dataset_name: str,
    data_dir: Path,
    split: str = 'train'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a FARE benchmark dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of the FARE dataset (e.g., 'ACSIncome-ALL-2014.pkl')
    data_dir : Path
        Directory containing the FARE datasets
    split : str, default='train'
        Which split to load: 'train', 'test', or 'val'
        
    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    s : np.ndarray
        Sensitive attributes
    """
    # Ensure dataset name has .pkl extension
    if not dataset_name.endswith('.pkl'):
        dataset_name = f'{dataset_name}.pkl'
    
    dataset_path = data_dir / dataset_name
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Load the dataset
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
        data = dict(data)
    
    # FARE datasets have structure: {'train': {...}, 'test': {...}, 'val': {...}}
    if split not in data:
        raise ValueError(f"Split '{split}' not found in dataset. Available: {list(data.keys())}")
    
    split_data = data[split]
    
    # Extract features, sensitive attribute, and labels
    X = split_data[0]  # Features
    s = split_data[1]  # Sensitive attribute
    y = split_data[2]  # Labels
    
    return X, y, s


def get_available_fare_datasets(data_dir: Path) -> List[str]:
    """
    Get list of available FARE datasets in the data directory.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing FARE datasets
        
    Returns
    -------
    datasets : List[str]
        List of dataset filenames (with .pkl extension)
    """
    if not data_dir.exists():
        return []
    
    datasets = [
        f for f in os.listdir(data_dir)
        if f.endswith('.pkl')
    ]
    
    return sorted(datasets)


def load_legacy_dataset(
    dataset_name: str,
    data_dir: Path = Path('data')
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load datasets in the legacy format (COMPAS, ADULT, LSAC).
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset: 'COMPAS', 'ADULT', or 'LSAC'
    data_dir : Path
        Base data directory
        
    Returns
    -------
    train : pd.DataFrame
        Training data with 'attr' and 'label' columns
    test : pd.DataFrame
        Test data
    val : pd.DataFrame
        Validation data
    """
    dataset_path = data_dir / dataset_name / 'data_file.pkl'
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        x1, x2, y1, y2 = pickle.load(f)
    
    # Create dataframes for each sensitive group
    if dataset_name in ['COMPAS', 'ADULT']:
        x1_df = pd.DataFrame(data=x1)
        x1_df['label'] = y1
        x1_df['attr'] = 0
        
        x2_df = pd.DataFrame(data=x2)
        x2_df['label'] = y2
        x2_df['attr'] = 1
        
    elif dataset_name == 'LSAC':
        x1_df = pd.DataFrame(data=x1)
        x1_df['label'] = (y1 > 3).astype(int)
        x1_df['attr'] = 0
        
        x2_df = pd.DataFrame(data=x2)
        x2_df['label'] = (y2 > 3).astype(int)
        x2_df['attr'] = 1
    else:
        raise ValueError(f"Unknown legacy dataset: {dataset_name}")
    
    # Combine and split
    data = pd.concat([x1_df, x2_df])
    
    from sklearn.model_selection import train_test_split
    
    # Create stratification column
    data['stratify_column'] = data['attr'].astype(str) + '_' + data['label'].astype(str)
    
    # Split into train/test/val
    train, temp = train_test_split(
        data, 
        test_size=0.3, 
        stratify=data['stratify_column'], 
        random_state=42
    )
    test, val = train_test_split(
        temp, 
        test_size=0.5, 
        stratify=temp['stratify_column'], 
        random_state=42
    )
    
    # Clean up
    train = train.drop('stratify_column', axis=1).reset_index(drop=True)
    test = test.drop('stratify_column', axis=1).reset_index(drop=True)
    val = val.drop('stratify_column', axis=1).reset_index(drop=True)
    
    return train, test, val


def print_dataset_info(X: np.ndarray, y: np.ndarray, s: np.ndarray, name: str = "Dataset"):
    """
    Print summary information about a dataset.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    s : np.ndarray
        Sensitive attributes
    name : str
        Name to display
    """
    print(f"\n{name} Information:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Sensitive attribute distribution: {np.bincount(s.astype(int))}")
    print(f"  Label distribution: {np.bincount(y.astype(int))}")
    print(f"  Missing values: {np.isnan(X).sum()}")
