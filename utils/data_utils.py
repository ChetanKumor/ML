import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

def load_dataset(file) -> pd.DataFrame:
    """
    Loads a CSV or Excel file from Streamlit's UploadedFile object or file path.
    Supports both .csv and .xlsx/.xls.
    """
    if hasattr(file, "name"):  # UploadedFile from Streamlit
        file_name = file.name
    elif isinstance(file, str):
        file_name = file
    else:
        raise ValueError("Invalid input type for file. Must be a path or UploadedFile.")

    if file_name.endswith(".csv"):
        return pd.read_csv(file)
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Only CSV and Excel are supported.")

def detect_target_column(df: pd.DataFrame) -> str:
    """
    Automatically selects the most likely target column by checking columns 
    with fewer unique values and suitable data types.
    """
    for col in reversed(df.columns):
        if df[col].nunique() < 20 and df[col].dtype in [object, int, float, bool]:
            return col
    return df.columns[-1]

def clean_currency_symbols(column: pd.Series) -> pd.Series:
    """
    Detects and removes currency symbols, percentages, or non-numeric characters 
    from numeric-looking columns.
    """
    return column.replace(r'[^0-9.-]', '', regex=True).astype(float)

def preprocess_target_column(y: pd.Series) -> Tuple[np.ndarray, object]:
    """
    Cleans and encodes the target column.
    - If it's numeric-looking with symbols, cleans and converts to float.
    - If it's categorical, encodes with LabelEncoder.
    """
    try:
        y_cleaned = clean_currency_symbols(y)
        return y_cleaned.values, None  # No encoder needed
    except:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        return y_encoded, le

def analyze_and_prepare_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, np.ndarray, object]:
    """
    Drops the target column from features, preprocesses it, and returns:
    - cleaned X (features)
    - cleaned y (target)
    - encoder used (or None)
    """
    y_raw = df[target_col]
    y, encoder = preprocess_target_column(y_raw)
    df = df.drop(columns=[target_col])
    return df, y, encoder
