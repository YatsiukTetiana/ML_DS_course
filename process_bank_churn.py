"""
process_bank_churn.py

Utility functions for preprocessing the Bank Customer Churn dataset.
"""

from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def split_train_validation(
    raw_df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and validation sets.

    Args:
        raw_df: Original dataframe.
        target_col: Name of the target column.
        test_size: Fraction of samples used for validation.
        random_state: Random seed.

    Returns:
        Training and validation dataframes.
    """
    return train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_df[target_col],
    )


def get_input_columns(
    df: pd.DataFrame,
    target_col: str,
) -> List[str]:
    """
    Determine model input columns.

    The Bank Churn dataset ignores the first two columns
    (RowNumber and CustomerId) and the target column.

    Args:
        df: Input dataframe.
        target_col: Name of target column.

    Returns:
        List of feature column names.
    """
    return [col for col in df.columns[2:] if col != target_col]


def split_inputs_targets(
    df: pd.DataFrame,
    input_cols: List[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataframe into model inputs and targets.

    Args:
        df: Dataset.
        input_cols: Feature column names.
        target_col: Target column name.

    Returns:
        Inputs dataframe and target series.
    """
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()

    return inputs, targets


def get_feature_types(
    inputs: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical feature columns.

    Args:
        inputs: Feature dataframe.

    Returns:
        Tuple containing numeric and categorical column names.
    """
    numeric_cols = inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = inputs.select_dtypes(include="object").columns.tolist()

    return numeric_cols, categorical_cols


def fit_scaler(
    train_inputs: pd.DataFrame,
    numeric_cols: List[str],
) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on the training numeric features.

    Args:
        train_inputs: Training feature dataframe.
        numeric_cols: Numeric column names.

    Returns:
        Fitted MinMaxScaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_inputs[numeric_cols])

    return scaler


def scale_features(
    inputs: pd.DataFrame,
    scaler: MinMaxScaler,
    numeric_cols: List[str],
) -> pd.DataFrame:
    """
    Scale numeric features.

    Args:
        inputs: Feature dataframe.
        scaler: Fitted scaler.
        numeric_cols: Numeric column names.

    Returns:
        Scaled dataframe.
    """
    inputs = inputs.copy()
    inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])

    return inputs


def fit_encoder(
    train_inputs: pd.DataFrame,
    categorical_cols: List[str],
) -> OneHotEncoder:
    """
    Fit a OneHotEncoder on categorical features.

    Args:
        train_inputs: Training feature dataframe.
        categorical_cols: Categorical column names.

    Returns:
        Fitted OneHotEncoder.
    """
    encoder = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore",
    )

    encoder.fit(train_inputs[categorical_cols])

    return encoder


def encode_features(
    inputs: pd.DataFrame,
    encoder: OneHotEncoder,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply one-hot encoding to categorical columns.

    Args:
        inputs: Feature dataframe.
        encoder: Fitted encoder.
        categorical_cols: Categorical column names.

    Returns:
        Updated dataframe and encoded column names.
    """
    inputs = inputs.copy()

    encoded_cols = list(
        encoder.get_feature_names_out(categorical_cols)
    )

    inputs[encoded_cols] = encoder.transform(inputs[categorical_cols])

    return inputs, encoded_cols


def build_feature_matrix(
    inputs: pd.DataFrame,
    numeric_cols: List[str],
    encoded_cols: List[str],
) -> pd.DataFrame:
    """
    Build the final model feature matrix.

    Args:
        inputs: Preprocessed dataframe.
        numeric_cols: Numeric columns.
        encoded_cols: One-hot encoded columns.

    Returns:
        Feature matrix.
    """
    return inputs[numeric_cols + encoded_cols]


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = True,
):
    """
    Complete preprocessing pipeline for the Bank Customer Churn dataset.

    The function:
        1. Splits the dataset.
        2. Separates inputs and targets.
        3. Fits a scaler on numeric features if scaler_numeric is True.
        4. Fits an encoder on categorical features.
        5. Applies scaling and encoding.
        6. Returns train/validation matrices and preprocessing objects.

    Args:
        raw_df: Raw Bank Customer Churn dataframe.
        scaler_numeric: Whether numeric features should be scaled using MinMaxScaler.
    Returns:
        (
            X_train,
            train_targets,
            X_val,
            val_targets,
            input_cols,
            scaler,
            encoder,
        )
    """
    target_col = "Exited"

    train_df, val_df = split_train_validation(raw_df, target_col)

    input_cols = get_input_columns(raw_df, target_col)

    train_inputs, train_targets = split_inputs_targets(
        train_df,
        input_cols,
        target_col,
    )

    val_inputs, val_targets = split_inputs_targets(
        val_df,
        input_cols,
        target_col,
    )

    numeric_cols, categorical_cols = get_feature_types(train_inputs)

    # Scale numeric features (optional)
    scaler = None

    if scaler_numeric:
        scaler = fit_scaler(train_inputs, numeric_cols)

        train_inputs = scale_features(
            train_inputs,
            scaler,
            numeric_cols,
        )

        val_inputs = scale_features(
            val_inputs,
            scaler,
            numeric_cols,
        )

    # Encode categorical features
    encoder = fit_encoder(
        train_inputs,
        categorical_cols,
    )

    train_inputs, encoded_cols = encode_features(
        train_inputs,
        encoder,
        categorical_cols,
    )

    val_inputs, _ = encode_features(
        val_inputs,
        encoder,
        categorical_cols,
    )

    # Build final matrices
    X_train = build_feature_matrix(
        train_inputs,
        numeric_cols,
        encoded_cols,
    )

    X_val = build_feature_matrix(
        val_inputs,
        numeric_cols,
        encoded_cols,
    )

    return (
        X_train,
        train_targets,
        X_val,
        val_targets,
        input_cols,
        scaler,
        encoder,
    )

def preprocess_new_data(
    input_df: pd.DataFrame,
    input_cols: List[str],
    scaler: MinMaxScaler,
    encoder: OneHotEncoder,
) -> pd.DataFrame:
    """
    Preprocess new data using an already fitted scaler and encoder.

    This function is intended for inference or evaluation on unseen data. 
    It applies the preprocessing objects learned
    from the training data without fitting them again.

    Args:
        input_df: New raw dataframe.
        input_cols: Feature columns used during training.
        scaler: Previously fitted MinMaxScaler. Can be None if scaling
            was disabled during training.
        encoder: Previously fitted OneHotEncoder.

    Returns:
        Preprocessed feature matrix ready for prediction.
    """
    inputs = input_df[input_cols].copy()

    numeric_cols, categorical_cols = get_feature_types(inputs)

    # Scale numeric features if a scaler was used during training
    if scaler is not None:
        inputs = scale_features(
            inputs,
            scaler,
            numeric_cols,
        )

    # Encode categorical features
    inputs, encoded_cols = encode_features(
        inputs,
        encoder,
        categorical_cols,
    )

    # Build final feature matrix
    X = build_feature_matrix(
        inputs,
        numeric_cols,
        encoded_cols,
    )

    return X