# app.py

import streamlit as st
import pandas as pd
import pickle
import os
import joblib # Import joblib for saving/loading
import numpy as np # Import numpy for np.nan
from utils.data_utils import load_dataset, analyze_and_prepare_target
from utils.feature_engineer import preprocess_data
from utils.model_trainer import train_models # This will be updated to return task_type
from utils.predict import make_prediction # Import the new make_prediction function
from utils.constants import MODEL_DIR, ENCODER_DIR
from utils.logging_utils import setup_logger
from datetime import datetime
import base64

logger = setup_logger("StreamlitApp")

st.set_page_config(page_title="Robo Data Scientist 🤖", layout="wide")
st.title("🤖 Robo Data Scientist - AutoML App")
st.markdown("Upload any structured dataset (CSV/Excel), and we'll train 15+ models, rank them, and let you make predictions!")

# Ensure MODEL_DIR and ENCODER_DIR exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)


# Sidebar for file upload and training
st.sidebar.header("📁 Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if file:
    df = load_dataset(file)
    st.subheader("📊 Raw Dataset Preview")
    st.dataframe(df.head())

    # Auto-detect target column or let user select
    from utils.data_utils import detect_target_column as auto_detect # Aliasing to avoid conflict
    initial_target_col_suggestion = auto_detect(df.copy()) # Pass a copy to avoid side effects

    # Find the index of the suggested target column
    try:
        default_index = list(df.columns).index(initial_target_col_suggestion)
    except ValueError:
        default_index = len(df.columns) - 1 if len(df.columns) > 0 else 0


    target_col = st.sidebar.selectbox(
        "🎯 Select Target Column",
        df.columns,
        index=default_index
    )
    st.sidebar.success(f"Selected target: {target_col}")

    if st.sidebar.button("🚀 Train Models"):
        with st.spinner("Training models... This may take a while! ⏳"):
            try:
                # Prepare data for training
                X_train_raw, y_train_processed, label_encoder = analyze_and_prepare_target(df.copy(), target_col)

                # Get task type using detect_task_type from auto_target_identifier
                from utils.auto_target_identifier import detect_task_type
                task_type = detect_task_type(pd.Series(y_train_processed)) # Pass a Series

                # Preprocess features using the pipeline
                X_processed_for_training, fitted_preprocessor_pipeline = preprocess_data(X_train_raw)

                # Combine processed features and target for model training
                processed_df_for_training = pd.DataFrame(X_processed_for_training)

                # Ensure column names are consistent if the preprocessor provides them
                try:
                    feature_names = fitted_preprocessor_pipeline.named_steps['column_transform'].get_feature_names_out()
                    processed_df_for_training.columns = feature_names
                except AttributeError:
                    logger.warning("ColumnTransformer does not support get_feature_names_out directly or it's not implemented for all steps. Using generic feature names.")
                    processed_df_for_training.columns = [f"feature_{i}" for i in range(X_processed_for_training.shape[1])]


                processed_df_for_training[target_col] = y_train_processed

                st.info("Training models... This might take a few moments.")
                # Pass task_type to train_models
                results, best_model_name, best_model_obj = train_models(processed_df_for_training.copy(), target_col, task_type)

                # Determine the primary metric for the leaderboard based on task type
                primary_metric_key = "Accuracy" if task_type == "classification" else "R2 Score"

                # Leaderboard
                st.subheader("🏆 Model Leaderboard")
                leaderboard_data = []
                for name, metrics_data in results.items():
                    leaderboard_entry = {
                        "Model": name,
                    }
                    
                    # Add primary metric from the top-level of metrics_data
                    if primary_metric_key in metrics_data and not pd.isna(metrics_data[primary_metric_key]):
                        leaderboard_entry[primary_metric_key] = round(metrics_data[primary_metric_key], 4)
                    else:
                        leaderboard_entry[primary_metric_key] = np.nan # Use NaN if primary metric is missing or NaN

                    # Add other relevant details from metrics_data['Details']
                    for detail_key, detail_value in metrics_data.get('Details', {}).items():
                        if isinstance(detail_value, (int, float)):
                            leaderboard_entry[detail_key] = round(detail_value, 4)
                        elif detail_key == "Confusion Matrix":
                            leaderboard_entry[detail_key] = str(detail_value) # Convert list to string for display
                        elif detail_key == "Error": # Include error message for failed models
                            leaderboard_entry[detail_key] = detail_value
                            
                    leaderboard_data.append(leaderboard_entry)

                # Always create and display the DataFrame.
                leaderboard_df = pd.DataFrame(leaderboard_data)

                if not leaderboard_df.empty:
                    # Sort by the primary metric if the column exists and has non-NaN values
                    if primary_metric_key in leaderboard_df.columns and not leaderboard_df[primary_metric_key].isnull().all():
                        ascending_sort = False # For R2 Score, higher is better, so descending values
                        if primary_metric_key == "Accuracy": # For accuracy, higher is better
                            ascending_sort = False
                        
                        # Sort, putting NaNs at the end
                        leaderboard_df = leaderboard_df.sort_values(
                            by=primary_metric_key, 
                            ascending=ascending_sort, 
                            na_position='last'
                        ).reset_index(drop=True)
                    st.dataframe(leaderboard_df)
                else:
                    st.info("No models were successfully trained or evaluated to display metrics.")


                # Conditional saving and download button
                if best_model_obj is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_filename = f"model_{best_model_name}_{timestamp}.pkl"
                    model_path = os.path.join(MODEL_DIR, model_filename)

                    # Save all necessary components together
                    joblib.dump((best_model_obj, fitted_preprocessor_pipeline, label_encoder, task_type), model_path)

                    st.success(f"Training complete! Best model '{best_model_name}' saved to '{model_path}'")

                    with open(model_path, "rb") as file_to_download:
                        btn = st.download_button(
                            label="📥 Download Best Model",
                            data=file_to_download,
                            file_name=os.path.basename(model_path),
                            mime="application/octet-stream"
                        )
                else:
                    st.warning("No best model could be trained successfully. Please check logs for errors or try a different dataset.")


            except Exception as e:
                st.error(f"Training failed: {e}")
                logger.error(f"Training Error: {e}", exc_info=True)
                st.exception(e) # Display full traceback in Streamlit

    # Prediction section
    st.sidebar.markdown("---")
    st.sidebar.header("🧪 Make Predictions")
    pred_file = st.sidebar.file_uploader("Upload Data for Prediction", type=["csv", "xlsx"], key="pred_uploader")

    # List available models
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    selected_model = st.sidebar.selectbox("Choose Saved Model", model_files if model_files else ["No model found"], key="model_selector")

    if pred_file and selected_model != "No model found":
        input_df_for_prediction = load_dataset(pred_file)
        model_path_for_prediction = os.path.join(MODEL_DIR, selected_model)

        with st.spinner("Making predictions..."):
            try:
                # Call the make_prediction function from utils.predict
                preds_df = make_prediction(input_df_for_prediction, model_path_for_prediction)
                st.subheader("🔮 Predictions")
                st.dataframe(preds_df)

                # Download predictions
                csv = preds_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions</a>'
                st.markdown(href, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                logger.error(f"Prediction Error: {e}", exc_info=True)
                st.exception(e) # Display full traceback in Streamlit
else:
    st.warning("Please upload a dataset to get started.")

