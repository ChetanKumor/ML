import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from typing import Tuple, Optional
from difflib import get_close_matches


def clean_numeric_string(val):
    if pd.isnull(val):
        return np.nan
    if isinstance(val, str):
        val = val.replace('$', '').replace('%', '').replace('₹', '').strip() # Added .strip()
        # Handle cases like "1,000.00" by removing commas
        val = val.replace(',', '')
        
        # If more than one decimal point, it's likely not a clean number
        if val.count('.') > 1:
            return np.nan
        
        # Remove any remaining non-numeric characters except for the first minus sign and one decimal point
        # This regex is more robust: ^-? captures optional leading minus, \d* captures digits,
        # (\.\d*)? captures optional decimal part, [^0-9.]* captures any other non-numeric non-dot characters at the end
        val = re.sub(r"([0-9.-]*[^0-9.])+", r"\1", val) # Remove multiple non-numeric chars
        
        try:
            return float(val)
        except ValueError:
            return np.nan
    return val


def clean_target_column(series: pd.Series) -> Tuple[np.ndarray, Optional[object]]:
    """
    Cleans and processes the target column.
    - Attempts to convert to numeric (float).
    - If not convertible to numeric, treats as categorical and applies LabelEncoder.
    Returns the processed target (np.ndarray) and the encoder (LabelEncoder or None).
    """
    # First, try to clean and convert to numeric.
    cleaned_numeric = series.apply(clean_numeric_string)

    # If the column *can* be treated as numeric (has valid numbers and not all NaNs)
    if pd.api.types.is_numeric_dtype(cleaned_numeric) and not cleaned_numeric.isnull().all():
        # Even if it looks like integers (e.g., 0.0, 1.0), we return it as numeric.
        # The detect_task_type function will decide if it's classification based on value properties.
        return cleaned_numeric.values, None
    else:
        # If it's truly non-numeric after cleaning (e.g., "A", "B", "C"),
        # then it's a categorical target that needs LabelEncoding.
        le = LabelEncoder()
        # Ensure conversion to string and fill NaNs for robust encoding
        y_encoded = le.fit_transform(series.astype(str).fillna('__MISSING__'))
        return y_encoded, le


def is_potential_target(series: pd.Series) -> bool:
    """
    Checks if a series is a potential target column based on basic heuristics.
    """
    if series.nunique() <= 1: # Too few unique values
        return False
    if series.isnull().mean() > 0.3: # Too many missing values
        return False
    if series.nunique() == len(series):  # All unique, likely an ID column
        return False
    return True


def detect_task_type(y_processed: pd.Series) -> str:
    """
    Detects if the task is classification or regression based on the PROCESSED target series.
    This function expects a series that has already gone through clean_target_column.

    A task is considered classification if:
    1. Its dtype is object, category, or bool (meaning it was string-like and encoded).
    2. It's numeric, but has a very small, discrete set of unique, integer-like values
       AND a limited range of values. This is a strict check to avoid misclassifying
       continuous numerical targets (like prices) as classification.
    """
    # If the processed series is categorical/boolean, it's definitely classification
    if y_processed.dtype in ['object', 'category', 'bool']:
        return 'classification'

    # If it's numerical, apply stricter checks
    if pd.api.types.is_numeric_dtype(y_processed):
        unique_non_nan = y_processed.dropna().unique()
        num_unique = len(unique_non_nan)

        # Calculate the range of unique numerical values
        # Handle cases where unique_non_nan might be empty or single value
        data_range = 0
        if num_unique > 1:
            data_range = unique_non_nan.max() - unique_non_nan.min()

        # Heuristic for numeric classification:
        # 1. Very few unique values (e.g., 2 for binary, up to ~10-15 for multi-class)
        # 2. ALL unique values must be integer-like (e.g., 0.0, 1.0, 2.0, but not 1.5)
        # 3. The range of these values must not be excessively large (e.g., prices vs. categories 0,1,2)

        # STRICT_NUMERIC_CLASSIFICATION_THRESHOLD: Max unique integer-like values for classification
        STRICT_NUMERIC_CLASSIFICATION_THRESHOLD = 15 # Can be adjusted based on typical classification class counts

        # ARBITRARY_MAX_CLASSIFICATION_VALUE: If the max value is too high, it's likely not classification
        # (e.g., a "class" value of 1000000 is usually a continuous variable)
        ARBITRARY_MAX_CLASSIFICATION_VALUE = 1000 # Example: if values are 0, 1, 2, 500, it might be OK.
                                                # If values are 10000, 20000, 30000, less likely classification.
                                                # This is a strong heuristic for 'price' data.

        # Combined condition:
        # - Has few unique values (and not empty)
        # - All unique values are integer-like
        # - Max value is within a reasonable range for discrete categories
        if (num_unique > 0 and num_unique <= STRICT_NUMERIC_CLASSIFICATION_THRESHOLD and
            np.all(np.mod(unique_non_nan, 1) == 0) and
            (unique_non_nan.max() <= ARBITRARY_MAX_CLASSIFICATION_VALUE or num_unique <= 5)): # Allow 5 classes regardless of max value (e.g. 100,200,300,400,500)
            return 'classification'
        
        # If it's numeric but doesn't meet the strict classification criteria, it's regression.
        return 'regression'

    # Default to regression if type is unexpected or not clearly classification
    return 'regression'


def auto_target_identifier(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Optional[object]]:
    """
    Automatically identifies the target column, preprocesses it, and returns
    features (X), processed target (y), and the encoder used.
    It also determines the task type based on the *processed* target.
    """
    # Prioritize columns with 'target', 'label' in name, or fewer unique values
    possible_names = ['target', 'label', 'class', 'output', 'y']
    target_col = None
    
    # Attempt 1: Fuzzy match on column names
    # Iterate through a copy of columns to allow modification during iteration if needed
    for col in list(df.columns):
        if get_close_matches(col.lower(), possible_names, n=1, cutoff=0.8):
            if is_potential_target(df[col]):
                target_col = col
                break
    
    # Attempt 2: Based on inferred task type from processed data
    # This loop will now use the stricter detect_task_type
    if not target_col:
        temp_target_suggestions = []
        for col in list(df.columns):
            if is_potential_target(df[col]):
                # Temporarily clean to make better task type decision
                temp_y_processed_array, _ = clean_target_column(df[col])
                temp_y_processed_series = pd.Series(temp_y_processed_array, index=df.index)
                
                # Check if this processed temporary series is classified as classification
                if detect_task_type(temp_y_processed_series) == 'classification':
                    # Store (number of unique values, column name) to prefer fewer classes
                    temp_target_suggestions.append((temp_y_processed_series.nunique(), col))
        
        if temp_target_suggestions:
            # Sort by number of unique values (fewer unique values usually means better classification target)
            temp_target_suggestions.sort()
            target_col = temp_target_suggestions[0][1] # Select the column with the fewest unique values

    # Attempt 3: Default to last column if no other strong candidate
    if not target_col and df.shape[1] >= 2:
        target_col = df.columns[-1]

    if not target_col:
        raise ValueError("❌ Could not auto-identify target column. Please ensure a suitable target column is present.")

    y_raw = df[target_col]
    X_df = df.drop(columns=[target_col])

    # Preprocess the target column
    y_processed_array, encoder = clean_target_column(y_raw)
    
    # Ensure y_processed is a Series for consistency in subsequent steps
    y_processed_series = pd.Series(y_processed_array, index=y_raw.index)

    return X_df, y_processed_series, encoder

