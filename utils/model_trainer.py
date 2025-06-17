# utils/model_trainer.py

import pandas as pd
import warnings
import numpy as np # Import numpy for sqrt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from utils.auto_target_identifier import detect_task_type # Import task type detector
from utils.logging_utils import get_logger # Import get_logger for consistent logging

logger = get_logger() # Initialize logger

warnings.filterwarnings("ignore")

def evaluate_model(model, X_test, y_test, task_type: str):
    """
    Evaluates a model based on its task type (classification or regression).

    Args:
        model: The trained model.
        X_test: Test features.
        y_test: True test labels/values.
        task_type (str): 'classification' or 'regression'.

    Returns:
        dict: A dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)

    if task_type == 'classification':
        # For classification, use average='weighted' for precision, recall, f1-score
        # to handle multi-class problems gracefully.
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist()
        }
    elif task_type == 'regression':
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse) # Calculate RMSE manually for broader compatibility
        return {
            "R2 Score": r2_score(y_test, y_pred),
            "Mean Squared Error": mse,
            "Root Mean Squared Error": rmse,
            "Mean Absolute Error": mean_absolute_error(y_test, y_pred)
            # Add other relevant regression metrics as needed
        }
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


# Define separate parameter grids and base models for classification and regression
CLASSIFICATION_PARAM_GRIDS = {
    "LogisticRegression": {"C": [0.01, 0.1, 1], "solver": ['liblinear']},
    "DecisionTree": {"max_depth": [None, 10], "min_samples_split": [2, 5]},
    "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
    "KNN": {"n_neighbors": [3, 5]},
    "NaiveBayes": {},
    "SVM": {"C": [0.1, 1], "kernel": ['linear', 'rbf']},
    "GradientBoosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
    "XGBoost": {"n_estimators": [50], "learning_rate": [0.01, 0.1], "use_label_encoder": [False], "eval_metric": ['logloss']},
    "LightGBM": {"n_estimators": [50], "learning_rate": [0.01, 0.1]},
    "CatBoost": {"depth": [4, 6], "learning_rate": [0.01, 0.1], "verbose": [0]},
}

CLASSIFICATION_BASE_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(probability=True, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

REGRESSION_PARAM_GRIDS = {
    "LinearRegression": {},
    "DecisionTreeRegressor": {"max_depth": [None, 10], "min_samples_split": [2, 5]},
    "RandomForestRegressor": {"n_estimators": [50, 100], "max_depth": [None, 10]},
    "KNeighborsRegressor": {"n_neighbors": [3, 5]},
    "SVR": {"C": [0.1, 1], "kernel": ['linear', 'rbf']},
    "GradientBoostingRegressor": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
    "XGBoostRegressor": {"n_estimators": [50], "learning_rate": [0.01, 0.1]},
    "LightGBMRegressor": {"n_estimators": [50], "learning_rate": [0.01, 0.1]},
    "CatBoostRegressor": {"depth": [4, 6], "learning_rate": [0.01, 0.1], "verbose": [0]},
}

REGRESSION_BASE_MODELS = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
    "RandomForestRegressor": RandomForestRegressor(random_state=42),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "SVR": SVR(),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    "XGBoostRegressor": XGBRegressor(random_state=42),
    "LightGBMRegressor": LGBMRegressor(random_state=42),
    "CatBoostRegressor": CatBoostRegressor(verbose=0, random_state=42)
}


def train_models(df: pd.DataFrame, target_col: str, task_type: str):
    """
    Trains multiple machine learning models and ensemble models,
    then evaluates and returns a leaderboard.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame including features and target.
        target_col (str): The name of the target column.
        task_type (str): 'classification' or 'regression'.

    Returns:
        Tuple[dict, str, object]: A tuple containing:
            - results (dict): Dictionary of model names and their evaluation metrics and model objects.
            - best_model_name (str): Name of the best performing model.
            - best_model_obj (object): The best performing model object.
            - task_type (str): The determined task type ('classification' or 'regression').
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"Detected task type: {task_type.upper()}")

    # Determine splitting strategy and initial best_score based on task type
    if task_type == 'classification':
        base_models_to_use = CLASSIFICATION_BASE_MODELS
        param_grids_to_use = CLASSIFICATION_PARAM_GRIDS
        scoring_metric = 'accuracy'
        
        # Check minimum class count for stratified split
        min_class_count = y.value_counts().min()
        if min_class_count >= 2: # Can stratify if at least 2 samples per class
            cv_splits = min(3, min_class_count) # Use min(3, actual_min_class_count) for robust CV
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
            # Only stratify if min_class_count is sufficient
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            logger.info(f"Using stratified train_test_split (min class count: {min_class_count}).")
        else: # Fallback to non-stratified if not enough data for stratification
            cv_splits = 2 # Default CV splits even without stratification
            cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42) # Use KFold for non-stratified
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logger.warning(f"Not enough data for stratified train_test_split (min class count: {min_class_count}). Using non-stratified split.")
        
        best_score = 0.0 # For accuracy, higher is better, start from a sensible low
    
    elif task_type == 'regression':
        base_models_to_use = REGRESSION_BASE_MODELS
        param_grids_to_use = REGRESSION_PARAM_GRIDS
        scoring_metric = 'r2' # Common scoring metric for regression
        cv = KFold(n_splits=5, shuffle=True, random_state=42) # Standard K-Fold for regression
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        best_score = -float('inf') # For R2 score, higher is better, but can be negative
    else:
        raise ValueError(f"Unsupported task type detected: {task_type}")


    results = {}
    best_model_name, best_model_obj = None, None


    logger.info("🔧 Training Models with Hyperparameter Tuning...")
    for name, model in base_models_to_use.items():
        logger.info(f"  Training {name}...")
        params = param_grids_to_use.get(name, {})
        
        # Adjust GridSearchCV estimator based on task type
        if params:
            grid_search_model = GridSearchCV(model, params, cv=cv, scoring=scoring_metric, n_jobs=-1, error_score='raise')
        else:
            grid_search_model = model # If no params, just use the model directly

        try:
            grid_search_model.fit(X_train, y_train)
            trained_model = grid_search_model.best_estimator_ if params else grid_search_model
            
            evaluation_metrics = evaluate_model(trained_model, X_test, y_test, task_type)
            score_key = "Accuracy" if task_type == 'classification' else "R2 Score"
            # Use .get with a default value to prevent KeyError if metric is missing
            score = evaluation_metrics.get(score_key, -float('inf') if task_type == 'regression' else 0.0)


            results[name] = {
                score_key: round(score, 4), # Store primary metric at top-level
                "Model": trained_model,
                "Details": evaluation_metrics
            }

            # Update best model only if the score is better
            if (task_type == 'classification' and score > best_score) or \
               (task_type == 'regression' and score > best_score):
                best_score, best_model_name, best_model_obj = score, name, trained_model
            logger.info(f"  {name} - Primary Metric ({score_key}): {score:.4f}")

        except Exception as e:
            logger.error(f"  Skipping {name} due to error: {e}", exc_info=True)
            # Add an entry to results even if training failed, to show in leaderboard
            results[name] = {
                ( "Accuracy" if task_type == 'classification' else "R2 Score"): np.nan, # Mark score as NaN
                "Model": None, # No trained model
                "Details": {"Error": str(e)} # Store error message
            }


    # Ensemble models (only for classification, extend for regression if needed)
    if task_type == 'classification':
        logger.info("\n🤝 Building Ensemble Models (Classification)...")

        # Ensure base models used in ensembles are the trained GridSearchCV estimators
        # or direct models if no tuning was done, and that they are not None (i.e., didn't fail)
        rf_model = results.get("RandomForest", {}).get("Model")
        xgb_model = results.get("XGBoost", {}).get("Model")
        lr_model = results.get("LogisticRegression", {}).get("Model")

        if rf_model and xgb_model and lr_model:
            try:
                voting = VotingClassifier(estimators=[
                    ('rf', rf_model),
                    ('xgb', xgb_model),
                    ('lr', lr_model)
                ], voting='soft', n_jobs=-1) # Add n_jobs for parallel processing
                voting.fit(X_train, y_train)
                evaluation_metrics = evaluate_model(voting, X_test, y_test, task_type)
                score = evaluation_metrics["Accuracy"]
                results["VotingEnsemble"] = {
                    "Accuracy": round(score, 4),
                    "Model": voting,
                    "Details": evaluation_metrics
                }
                if score > best_score:
                    best_score, best_model_name, best_model_obj = score, "VotingEnsemble", voting
                logger.info(f"  VotingEnsemble - Accuracy: {score:.4f}")
            except Exception as e:
                logger.error(f"  Skipping VotingEnsemble due to error: {e}", exc_info=True)
                results["VotingEnsemble"] = {
                    "Accuracy": np.nan,
                    "Model": None,
                    "Details": {"Error": str(e)}
                }
        else:
            logger.info("  Not enough base models available for VotingEnsemble or some failed.")
            results["VotingEnsemble"] = {
                "Accuracy": np.nan,
                "Model": None,
                "Details": {"Error": "Not enough successful base models to build ensemble."}
            }


        svm_model = results.get("SVM", {}).get("Model")
        lgbm_model = results.get("LightGBM", {}).get("Model")
        knn_model = results.get("KNN", {}).get("Model")

        if svm_model and lgbm_model and knn_model:
            try:
                stacking = StackingClassifier(
                    estimators=[
                        ('svc', svm_model),
                        ('lgbm', lgbm_model),
                        ('knn', knn_model)
                    ],
                    final_estimator=LogisticRegression(random_state=42),
                    n_jobs=-1 # Add n_jobs for parallel processing
                )
                stacking.fit(X_train, y_train)
                evaluation_metrics = evaluate_model(stacking, X_test, y_test, task_type)
                score = evaluation_metrics["Accuracy"]
                results["StackingEnsemble"] = {
                    "Accuracy": round(score, 4),
                    "Model": stacking,
                    "Details": evaluation_metrics
                }
                if score > best_score:
                    best_score, best_model_name, best_model_obj = score, "StackingEnsemble", stacking
                logger.info(f"  StackingEnsemble - Accuracy: {score:.4f}")
            except Exception as e:
                logger.error(f"  Skipping StackingEnsemble due to error: {e}", exc_info=True)
                results["StackingEnsemble"] = {
                    "Accuracy": np.nan,
                    "Model": None,
                    "Details": {"Error": str(e)}
                }
        else:
            logger.info("  Not enough base models available for StackingEnsemble or some failed.")
            results["StackingEnsemble"] = {
                "Accuracy": np.nan,
                "Model": None,
                "Details": {"Error": "Not enough successful base models to build ensemble."}
            }
    elif task_type == 'regression':
        # Add regression ensemble models here if desired, similar to classification
        logger.info("\n🤝 Ensemble Models for Regression are not yet implemented.")
        # Add placeholder entries for regression ensembles if they are not implemented,
        # so they still appear in the leaderboard with NaN values.
        results["RegressionVotingEnsemble"] = {
            "R2 Score": np.nan,
            "Model": None,
            "Details": {"Error": "Ensemble not implemented for regression."}
        }
        results["RegressionStackingEnsemble"] = {
            "R2 Score": np.nan,
            "Model": None,
            "Details": {"Error": "Ensemble not implemented for regression."}
        }


    logger.info("\n✅ All models trained.")
    return results, best_model_name, best_model_obj

