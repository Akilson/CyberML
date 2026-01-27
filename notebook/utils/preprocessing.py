import pandas as pd
import numpy as np
from typing import Tuple


def fill_missing_values(
    df: pd.DataFrame,
    numerical_fill: float = 0.0,
    categorical_fill: str = "Unknown"
) -> pd.DataFrame:
    """
    Fill missing values in a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with potential missing values
    numerical_fill : float
        Value to fill for numerical columns (default: 0.0)
    categorical_fill : str
        Value to fill for categorical/string columns (default: "Unknown")
        
    Returns
    -------
    pd.DataFrame
        Dataframe with missing values filled
    """
    df_filled = df.copy()

    for col in df_filled.columns:
        if df_filled[col].isna().any():
            if pd.api.types.is_numeric_dtype(df_filled[col]):
                df_filled[col] = df_filled[col].fillna(numerical_fill)
            else:
                df_filled[col] = df_filled[col].fillna(categorical_fill)

    return df_filled


def balance_dataset(
    df: pd.DataFrame,
    label_column: str = "Main_Label",
    target_total: int = 500_000,
    random_state: int = 42,
    fill_na: bool = True
) -> pd.DataFrame:
    """
    Balance a dataset by combining undersampling and oversampling strategies.
    
    For classes with more samples than target: undersample
    For classes with fewer samples than target: oversample (with replacement)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with imbalanced classes
    label_column : str
        Name of the column containing class labels
    target_total : int
        Target total number of samples in the balanced dataset
    random_state : int
        Random seed for reproducibility
    fill_na : bool
        Whether to fill missing values before balancing (default: True)
        
    Returns
    -------
    pd.DataFrame
        Balanced dataframe with approximately equal class distribution
    """
    np.random.seed(random_state)

    if fill_na:
        df = fill_missing_values(df)

    classes = df[label_column].unique()
    n_classes = len(classes)

    samples_per_class = target_total // n_classes

    balanced_dfs = []

    for cls in classes:
        class_df = df[df[label_column] == cls]
        current_count = len(class_df)

        if current_count >= samples_per_class:
            sampled_df = class_df.sample(n=samples_per_class, random_state=random_state)
        else:
            sampled_df = class_df.sample(n=samples_per_class, replace=True, random_state=random_state)
        
        balanced_dfs.append(sampled_df)

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return balanced_df


def get_class_distribution(df: pd.DataFrame, label_column: str = "Main_Label") -> pd.DataFrame:
    """
    Get the class distribution of a dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    label_column : str
        Name of the column containing class labels
        
    Returns
    -------
    pd.DataFrame
        Dataframe with class counts and percentages
    """
    counts = df[label_column].value_counts()
    percentages = df[label_column].value_counts(normalize=True) * 100

    distribution = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(2)
    })

    return distribution.sort_values("count", ascending=False)


def preprocess_datasets(
    flow_data_path: str,
    packet_data_path: str,
    target_total: int = 500_000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess both flow and packet datasets.
    
    Parameters
    ----------
    flow_data_path : str
        Path to flow_data.pkl file
    packet_data_path : str
        Path to packet_data.pkl file
    target_total : int
        Target total number of samples per dataset
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing (balanced_flow_data, balanced_packet_data)
    """
    flow_data = pd.read_pickle(flow_data_path)
    packet_data = pd.read_pickle(packet_data_path)

    balanced_flow = balance_dataset(flow_data, target_total=target_total, random_state=random_state)
    balanced_packet = balance_dataset(packet_data, target_total=target_total, random_state=random_state)

    return balanced_flow, balanced_packet


def save_balanced_datasets(
    balanced_flow: pd.DataFrame,
    balanced_packet: pd.DataFrame,
    output_dir: str
) -> Tuple[str, str]:
    """
    Save balanced datasets to pickle files.
    
    Parameters
    ----------
    balanced_flow : pd.DataFrame
        Balanced flow dataset
    balanced_packet : pd.DataFrame
        Balanced packet dataset
    output_dir : str
        Directory to save the files
        
    Returns
    -------
    Tuple[str, str]
        Paths to the saved files
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    flow_path = os.path.join(output_dir, "flow_data_balanced.pkl")
    packet_path = os.path.join(output_dir, "packet_data_balanced.pkl")

    balanced_flow.to_pickle(flow_path)
    balanced_packet.to_pickle(packet_path)

    return flow_path, packet_path
