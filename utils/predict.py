# utils/predict.py

import pandas as pd
import pickle
import joblib # Using joblib for saving/loading models and pipelines
import os
from utils.constants import MODEL_DIR # Assuming MODEL_DIR is defined in constants.py
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Set up a logger for the predict module
from utils.logging_utils import setup_logger
logger = setup_logger("PredictModule")


def make_prediction(input_df: pd.DataFrame, model_filepath: str) -> pd.DataFrame:
    """
    Loads a saved model, preprocessor, and label encoder, then makes predictions
    on the input DataFrame.

    Args:
        input_df (pd.DataFrame): The raw input data for which to make predictions.
                                 Must have the same column structure as the data
                                 used for training, excluding the target column.
        model_filepath (str): The full path to the .pkl file containing the
                              saved (model, preprocessor, encoder) tuple.

    Returns:
        pd.DataFrame: A DataFrame with the original input data and a new
                      'Predicted_Target' column.

    Raises:
        ValueError: If the model file cannot be loaded or is corrupted.
        Exception: For any other errors during prediction.
    """
    if not os.path.exists(model_filepath):
        logger.error(f"Model file not found: {model_filepath}")
        raise FileNotFoundError(f"Model file not found at {model_filepath}")

    try:
        # Load the saved tuple: (best_model_obj, preprocessor, encoder)
        # Using joblib for better compatibility with scikit-learn objects
        best_model, preprocessor, label_encoder = joblib.load(model_filepath)
        logger.info(f"Successfully loaded model, preprocessor, and label encoder from {model_filepath}")
    except Exception as e:
        logger.error(f"Error loading model components from {model_filepath}: {e}")
        raise ValueError(f"Failed to load model components. Check file integrity: {e}")

    # Make a copy to avoid modifying the original input_df
    data_to_predict = input_df.copy()

    try:
        # Apply the preprocessor pipeline to the new input data
        # Ensure the input to preprocessor.transform is a pandas DataFrame,
        # as the custom transformers in feature_engineer.py expect DataFrames.
        logger.info(f"Applying preprocessor to input data. Input shape: {data_to_predict.shape}")
        # The preprocessor pipeline (from PreprocessorBuilder) expects a DataFrame
        # and returns a NumPy array after the ColumnTransformer step.
        processed_data_for_prediction = preprocessor.transform(data_to_predict)
        logger.info(f"Preprocessing complete. Processed data shape: {processed_data_for_prediction.shape}")

        # Make predictions using the loaded model
        logger.info("Making predictions...")
        predictions_encoded = best_model.predict(processed_data_for_prediction)
        logger.info("Predictions made.")

        # Inverse transform if a label encoder was used (for classification tasks)
        if label_encoder is not None and isinstance(label_encoder, LabelEncoder):
            logger.info("Inverse transforming predictions using LabelEncoder.")
            predictions_decoded = label_encoder.inverse_transform(predictions_encoded)
        else:
            # If no LabelEncoder was used (e.g., regression or already numeric targets)
            predictions_decoded = predictions_encoded

        # Add predictions to a copy of the original input DataFrame
        result_df = input_df.copy()
        result_df['Predicted_Target'] = predictions_decoded
        logger.info("Predictions added to the result DataFrame.")

        return result_df

    except Exception as e:
        logger.error(f"Error during prediction process: {e}", exc_info=True)
        raise Exception(f"Prediction failed: {e}")

