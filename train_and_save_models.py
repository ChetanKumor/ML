# train_and_save.py

import pandas as pd
import joblib
import os
from utils.model_trainer import train_models
from utils.feature_engineer import preprocess_data
from utils.auto_target_identifier import identify_target_column
from utils.logging_utils import log
from sklearn.preprocessing import LabelEncoder

def main_train_pipeline(file_path, model_save_path="models/best_model.pkl"):
    log("📥 Loading dataset...")
    df = pd.read_csv(file_path)

    log("🔍 Auto-identifying target column...")
    target_col = identify_target_column(df)
    log(f"🎯 Target column identified: {target_col}")

    log("🧼 Preprocessing features...")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
        df[target_col] = y
        joblib.dump(le, "models/label_encoder.pkl")
        log("🔡 Target column encoded using LabelEncoder.")

    X_processed, preprocessor = preprocess_data(X)
    df_processed = X_processed.copy()
    df_processed[target_col] = y

    log("🤖 Training models...")
    results, best_model_name, best_model_obj = train_models(df_processed, target_col)

    log(f"✅ Best model: {best_model_name} with accuracy {results[best_model_name]['Accuracy']}")

    log("💾 Saving best model and preprocessor...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model_obj, model_save_path)
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    log("📊 Training complete. Models and pipeline saved.")
    return results, best_model_name

if __name__ == "__main__":
    results, best_model = main_train_pipeline("data/sample.csv")
