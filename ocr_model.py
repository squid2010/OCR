# ocr_model.py
# CNN-LSTM-CTC OCR model for handwritten medical text recognition
# Refactored to use functional API and custom CTC loss (no Lambda layer)

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

def build_ocr_model(
    img_height=128,
    img_width=384,
    num_chars=80,
    max_text_length=32,
    lstm_units=256,
    dropout_rate=0.25
):
    # Inputs
    image_input = layers.Input(shape=(img_height, img_width, 1), name='image', dtype='float32')


    # CNN feature extractor
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(image_input)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(dropout_rate)(x)

    # Reshape for RNN
    # After CNN stack, x.shape = (batch_size, 4, 24, 256)
    # Reshape to (batch_size, 24, 4*256) for RNN input
    x = layers.Reshape((24, 4 * 256))(x)

    # RNN layers
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=0.25))(x)

    # Output
    y_pred = layers.Dense(num_chars + 1, activation='softmax', dtype='float32')(x)  # +1 for CTC blank

    # Model for training and prediction (single input)
    model = tf.keras.models.Model(inputs=image_input, outputs=y_pred)
    prediction_model = model

    return model, prediction_model

# =========================
# CTC Decoding and Char Mapping Utilities
# =========================

import numpy as np
import pickle

def decode_predictions(preds, num_to_char, input_lengths=None):
    """
    Decodes a batch of CTC predictions to strings.
    Args:
        preds: numpy array (batch, time_steps, num_classes)
        num_to_char: dict mapping int to char
        input_lengths: optional, lengths of each prediction (if None, use preds.shape[1])
    Returns:
        List of decoded strings
    """
    from tensorflow.keras import backend as K
    if input_lengths is None:
        input_lengths = np.ones(preds.shape[0]) * preds.shape[1]
    decoded, _ = K.ctc_decode(preds, input_length=input_lengths, greedy=True)
    decoded = K.get_value(decoded[0])
    results = []
    for seq in decoded:
        text = ''.join([num_to_char.get(i, '') for i in seq if i != -1])
        results.append(text)
    return results

def load_char_mappings(path):
    """
    Loads char_to_num and num_to_char from a pickle file.
    """
    with open(path, "rb") as f:
        mappings = pickle.load(f)
    return mappings["char_to_num"], mappings["num_to_char"]

# Utility for Apple Silicon: limit memory growth (prevents OOM on M1/M2)
def set_memory_growth():
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# Example usage (for script or notebook)
if __name__ == "__main__":
    set_memory_growth()
    # Example: build model for 80-character vocab
    model, pred_model = build_ocr_model(
        img_height=128,
        img_width=384,
        num_chars=80,
        max_text_length=32,
        lstm_units=256
    )
    model.summary()

# =========================
# CTC Decoding and Char Mapping Utilities
# =========================

import numpy as np
import pickle

def decode_predictions(preds, num_to_char, input_lengths=None):
    """
    Decodes a batch of CTC predictions to strings.
    Args:
        preds: numpy array (batch, time_steps, num_classes)
        num_to_char: dict mapping int to char
        input_lengths: optional, lengths of each prediction (if None, use preds.shape[1])
    Returns:
        List of decoded strings
    """
    from tensorflow.keras import backend as K
    if input_lengths is None:
        input_lengths = np.ones(preds.shape[0]) * preds.shape[1]
    decoded, _ = K.ctc_decode(preds, input_length=input_lengths, greedy=True)
    decoded = K.get_value(decoded[0])
    results = []
    for seq in decoded:
        text = ''.join([num_to_char.get(i, '') for i in seq if i != -1])
        results.append(text)
    return results

def load_char_mappings(path):
    """
    Loads char_to_num and num_to_char from a pickle file.
    """
    with open(path, "rb") as f:
        mappings = pickle.load(f)
    return mappings["char_to_num"], mappings["num_to_char"]

# Utility for Apple Silicon: limit memory growth (prevents OOM on M1/M2)
def set_memory_growth():
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# Example usage (for script or notebook)
if __name__ == "__main__":
    set_memory_growth()
    # Example: build model for 80-character vocab
    model, pred_model = build_ocr_model(
        img_height=128,
        img_width=384,
        num_chars=80,
        max_text_length=32,
        lstm_units=256
    )
    model.summary()
