import os

# === Base Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

MODEL_DIR = os.path.join(PARENT_DIR, "saved_models")
ENCODER_DIR = os.path.join(PARENT_DIR, "saved_encoders")
LOG_DIR = os.path.join(PARENT_DIR, "logs")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === File Paths ===
PREPROCESSOR_SAVE_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
LEADERBOARD_SAVE_PATH = os.path.join(MODEL_DIR, "leaderboard.json")
LOG_FILE_PATH = os.path.join(LOG_DIR, "app.log")

# === File Limits ===
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = [".csv"]

# === App Info ===
APP_TITLE = "🤖 Robo Data Scientist"
APP_DESCRIPTION = "Upload your CSV, train ML models, get predictions, and see model leaderboard automatically."

# === ML Config ===
DEFAULT_SCORING = "accuracy"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

# === Metrics ===
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1 Score"]

# === Color Map ===
COLOR_MAP = {
    "Accuracy": "#2ECC71",
    "Precision": "#3498DB",
    "Recall": "#F39C12",
    "F1 Score": "#E74C3C"
}
