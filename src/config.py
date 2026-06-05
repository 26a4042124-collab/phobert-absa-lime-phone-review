from pathlib import Path

# =========================
# PATH CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR = BASE_DIR / "models" / "phobert_absa"

OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

# =========================
# MODEL CONFIG
# =========================

MODEL_NAME = "vinai/phobert-base"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# Nếu máy yếu hoặc muốn test nhanh, đổi DEBUG = True
DEBUG = False
DEBUG_SAMPLE_SIZE = 3000
EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 128

# =========================
# LABEL CONFIG
# =========================

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}