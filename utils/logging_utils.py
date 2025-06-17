# utils/logging_utils.py

import os
import sys
import logging
from datetime import datetime

def setup_logger(name: str = "robo_logger", log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent adding duplicate handlers in case of Streamlit reloads
    if logger.handlers:
        return logger

    # --- File Handler (UTF-8 safe) ---
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(file_format)

    # --- Console Handler (UTF-8 safe) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)

    # --- Add handlers ---
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 🚀 Emoji-safe log message
    logger.info(f"🚀 Logger initialized at {log_path}")
    return logger


# Global logger getter (singleton style)
_robo_logger = None

def get_logger():
    global _robo_logger
    if _robo_logger is None:
        _robo_logger = setup_logger()
    return _robo_logger
