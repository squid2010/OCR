# Configuration settings for Medical Handwriting OCR System
# Optimized for Mac M1 (Apple Silicon) and memory safety

import platform

class OCRConfig:
    # Image dimensions (should match dataset)
    IMG_HEIGHT = 128
    IMG_WIDTH = 384

    # Maximum length of text labels (characters)
    MAX_TEXT_LENGTH = 32

    # Training parameters
    BATCH_SIZE = 1  # Lowered for Mac M1 and to avoid OOM
    EPOCHS = 21
    LEARNING_RATE = 0.001

    # Model architecture
    LSTM_UNITS = 256
    CNN_FILTERS = [32, 64, 128, 128, 256]
    DROPOUT_RATE = 0.25

    # Data augmentation
    USE_AUGMENTATION = True

    # Mixed precision (Apple Silicon supports float16, but can be disabled if issues)
    USE_MIXED_PRECISION = True

    # Early stopping and LR reduction
    EARLY_STOPPING_PATIENCE = 8
    LR_REDUCE_PATIENCE = 4
    LR_REDUCE_FACTOR = 0.5
    MIN_LR = 1e-6

    # Checkpointing
    SAVE_BEST_ONLY = True

    # Hardware/acceleration
    USE_XLA = False  # XLA is not always stable on Mac, set to True if you want to experiment

    # Data paths
    TRAIN_DIR = "trainingData/Training/training_words"
    TRAIN_LABELS = "trainingData/Training/training_labels.csv"
    VAL_DIR = "trainingData/Validation/validation_words"
    VAL_LABELS = "trainingData/Validation/validation_labels.csv"
    TEST_DIR = "trainingData/Testing/testing_words"
    TEST_LABELS = "trainingData/Testing/testing_labels.csv"

    # Output/model paths
    MODEL_DIR = "models"
    BEST_MODEL_PATH = f"{MODEL_DIR}/ocr_model_best.h5"
    FINAL_MODEL_PATH = f"{MODEL_DIR}/ocr_model_final.h5"
    PREDICTION_MODEL_PATH = f"{MODEL_DIR}/ocr_model_prediction.keras"
    CHAR_MAPPINGS_PATH = f"{MODEL_DIR}/char_mappings.pkl"
    TRAINING_HISTORY_PLOT = "training_history.png"
    DATA_DIR = "data"

    # Evaluation output
    VAL_REPORT = "validation_evaluation_report.txt"
    TEST_REPORT = "test_evaluation_report.txt"
    VAL_PLOTS = "validation_evaluation_plots.png"
    TEST_PLOTS = "test_evaluation_plots.png"
    VAL_PRED_CSV = "validation_prediction_comparison.csv"
    TEST_PRED_CSV = "test_prediction_comparison.csv"

    # Mac M1 specific tweaks
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        # Apple Silicon: Use smaller batch, mixed precision, and avoid XLA by default
        BATCH_SIZE = 8
        USE_MIXED_PRECISION = True
        USE_XLA = False

class LightweightConfig(OCRConfig):
    # Smaller model for faster training/less memory
    LSTM_UNITS = 128
    CNN_FILTERS = [16, 32, 64, 64, 128]
    BATCH_SIZE = 16
    EPOCHS = 30

class HighAccuracyConfig(OCRConfig):
    # Larger model for higher accuracy
    LSTM_UNITS = 384
    CNN_FILTERS = [64, 128, 256, 256, 512]
    BATCH_SIZE = 4
    EPOCHS = 80
    DROPOUT_RATE = 0.3

class FastTrainingConfig(OCRConfig):
    # For quick experimentation
    LSTM_UNITS = 64
    CNN_FILTERS = [16, 32, 64, 64, 128]
    BATCH_SIZE = 32
    EPOCHS = 10
    USE_AUGMENTATION = False

# For easy import
__all__ = [
    "OCRConfig",
    "LightweightConfig",
    "HighAccuracyConfig",
    "FastTrainingConfig"
]
