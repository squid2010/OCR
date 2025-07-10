import os

class OCRConfig:
    """Configuration class for OCR model parameters"""

    # Image dimensions
    IMG_HEIGHT = 128
    IMG_WIDTH = 384
    IMG_CHANNELS = 1

    # Model parameters
    MAX_TEXT_LENGTH = 32
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # Training parameters
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LEARNING_RATE = 1e-7

    # Data augmentation parameters
    USE_DATA_AUGMENTATION = True
    ROTATION_RANGE = 5
    WIDTH_SHIFT_RANGE = 0.1
    HEIGHT_SHIFT_RANGE = 0.1
    ZOOM_RANGE = 0.1
    BRIGHTNESS_RANGE = [0.8, 1.2]

    # Model architecture parameters
    CNN_FILTERS = [32, 64, 128, 128, 256]
    LSTM_UNITS = 256
    DROPOUT_RATE = 0.2
    USE_BIDIRECTIONAL = True
    USE_BATCH_NORMALIZATION = True

    # CTC parameters
    CTC_MERGE_REPEATED = True

    # Paths
    DATA_DIR = "trainingData"
    TRAIN_DIR = os.path.join(DATA_DIR, "Training")
    VAL_DIR = os.path.join(DATA_DIR, "Validation")
    TEST_DIR = os.path.join(DATA_DIR, "Testing")

    # Image subdirectories
    TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, "training_words")
    VAL_IMG_DIR = os.path.join(VAL_DIR, "validation_words")
    TEST_IMG_DIR = os.path.join(TEST_DIR, "testing_words")

    TRAIN_CSV = os.path.join(TRAIN_DIR, "training_labels.csv")
    VAL_CSV = os.path.join(VAL_DIR, "validation_labels.csv")
    TEST_CSV = os.path.join(TEST_DIR, "testing_labels.csv")

    # Model save paths
    MODEL_DIR = "models"
    MODEL_NAME = "ocr_model"
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.h5")
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_final.h5")
    PREDICTION_MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_prediction.h5")
    CHAR_MAPPINGS_PATH = os.path.join(MODEL_DIR, "char_mappings.pkl")

    # Logging and visualization
    LOG_DIR = "logs"
    PLOTS_DIR = "plots"
    TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, "tensorboard")

    # Evaluation parameters
    CONFIDENCE_THRESHOLD = 0.5
    MAX_EDIT_DISTANCE = 10

    # Hardware optimization
    USE_MIXED_PRECISION = True
    USE_XLA = True

    # Preprocessing parameters
    NORMALIZE_PIXEL_VALUES = True
    PIXEL_MEAN = 0.5
    PIXEL_STD = 0.5

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.MODEL_DIR,
            cls.LOG_DIR,
            cls.PLOTS_DIR,
            cls.TENSORBOARD_LOG_DIR
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def get_model_config(cls):
        """Return model configuration as dictionary"""
        return {
            'img_height': cls.IMG_HEIGHT,
            'img_width': cls.IMG_WIDTH,
            'img_channels': cls.IMG_CHANNELS,
            'max_length': cls.MAX_TEXT_LENGTH,
            'batch_size': cls.BATCH_SIZE,
            'epochs': cls.EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'cnn_filters': cls.CNN_FILTERS,
            'lstm_units': cls.LSTM_UNITS,
            'dropout_rate': cls.DROPOUT_RATE,
            'use_bidirectional': cls.USE_BIDIRECTIONAL,
            'use_batch_norm': cls.USE_BATCH_NORMALIZATION
        }

    @classmethod
    def get_training_config(cls):
        """Return training configuration as dictionary"""
        return {
            'validation_split': cls.VALIDATION_SPLIT,
            'early_stopping_patience': cls.EARLY_STOPPING_PATIENCE,
            'reduce_lr_patience': cls.REDUCE_LR_PATIENCE,
            'reduce_lr_factor': cls.REDUCE_LR_FACTOR,
            'min_learning_rate': cls.MIN_LEARNING_RATE,
            'use_mixed_precision': cls.USE_MIXED_PRECISION,
            'use_xla': cls.USE_XLA
        }

    @classmethod
    def get_augmentation_config(cls):
        """Return data augmentation configuration as dictionary"""
        return {
            'use_augmentation': cls.USE_DATA_AUGMENTATION,
            'rotation_range': cls.ROTATION_RANGE,
            'width_shift_range': cls.WIDTH_SHIFT_RANGE,
            'height_shift_range': cls.HEIGHT_SHIFT_RANGE,
            'zoom_range': cls.ZOOM_RANGE,
            'brightness_range': cls.BRIGHTNESS_RANGE
        }

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("OCR Model Configuration:")
        print("=" * 50)
        print(f"Image dimensions: {cls.IMG_HEIGHT}x{cls.IMG_WIDTH}x{cls.IMG_CHANNELS}")
        print(f"Max text length: {cls.MAX_TEXT_LENGTH}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"LSTM units: {cls.LSTM_UNITS}")
        print(f"Dropout rate: {cls.DROPOUT_RATE}")
        print(f"Use bidirectional LSTM: {cls.USE_BIDIRECTIONAL}")
        print(f"Use batch normalization: {cls.USE_BATCH_NORMALIZATION}")
        print(f"Use data augmentation: {cls.USE_DATA_AUGMENTATION}")
        print(f"Use mixed precision: {cls.USE_MIXED_PRECISION}")
        print("=" * 50)

# Alternative configurations for different scenarios
class LightweightConfig(OCRConfig):
    """Lightweight configuration for faster training/inference"""
    CNN_FILTERS = [16, 32, 64, 64, 128]
    LSTM_UNITS = 128
    BATCH_SIZE = 32
    EPOCHS = 30

class HighAccuracyConfig(OCRConfig):
    """High accuracy configuration with more parameters"""
    CNN_FILTERS = [64, 128, 256, 256, 512]
    LSTM_UNITS = 512
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 0.0005
    DROPOUT_RATE = 0.3

class FastTrainingConfig(OCRConfig):
    """Configuration optimized for fast training"""
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.002
    EARLY_STOPPING_PATIENCE = 5
    REDUCE_LR_PATIENCE = 3
