# ocr_model.py
# CNN-LSTM-CTC OCR model for handwritten medical text recognition
# Compatible with Mac M1 (Apple Silicon) and optimized for memory safety

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# Mixed precision policy for Apple Silicon (M1/M2) - safe fallback to float32 if not available
try:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
except Exception:
    # Fallback for older TF or non-mixed precision environments
    pass

class OCRModel(tf.keras.Model):
    def __init__(
        self,
        img_height=128,
        img_width=384,
        num_chars=80,
        max_text_length=32,
        lstm_units=256,
        dropout_rate=0.25,
        **kwargs
    ):
        super(OCRModel, self).__init__(**kwargs)
        self.img_height = img_height
        self.img_width = img_width
        self.num_chars = num_chars
        self.max_text_length = max_text_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate

        # Feature extractor (CNN)
        self.cnn = models.Sequential([
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),

            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),

            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),

            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),

            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 1)),  # Only downsample height
            layers.BatchNormalization(),

            layers.Dropout(self.dropout_rate),
        ])

        # Sequence modeling (LSTM)
        self.reshape = layers.Reshape(target_shape=(-1, 256))  # (batch, time_steps, features)
        self.bilstm1 = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True, dropout=0.25))
        self.bilstm2 = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True, dropout=0.25))

        # Output layer (CTC)
        self.dense = layers.Dense(self.num_chars + 1, activation='softmax', dtype='float32')  # +1 for CTC blank

    def call(self, inputs, training=False):
        x = self.cnn(inputs, training=training)
        x = self.reshape(x)
        x = self.bilstm1(x, training=training)
        x = self.bilstm2(x, training=training)
        x = self.dense(x)
        return x

def build_ocr_model(
    img_height=128,
    img_width=384,
    num_chars=80,
    max_text_length=32,
    lstm_units=256,
    dropout_rate=0.25
):
    """
    Returns a compiled OCR model (Keras Model) with CTC loss.
    """
    # Input layers
    image_input = layers.Input(shape=(img_height, img_width, 1), name='image', dtype='float32')
    labels = layers.Input(name='label', shape=(max_text_length,), dtype='int32')
    input_length = layers.Input(name='input_length', shape=(1,), dtype='int32')
    label_length = layers.Input(name='label_length', shape=(1,), dtype='int32')

    # Model body
    ocr_body = OCRModel(
        img_height=img_height,
        img_width=img_width,
        num_chars=num_chars,
        max_text_length=max_text_length,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate
    )
    y_pred = ocr_body(image_input)

    # CTC loss as a Lambda layer
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        # y_pred is (batch, time_steps, num_classes)
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    loss_out = layers.Lambda(ctc_lambda_func, name='ctc_loss')([y_pred, labels, input_length, label_length])

    # Model for training (with CTC loss)
    model = tf.keras.models.Model(
        inputs=[image_input, labels, input_length, label_length],
        outputs=loss_out
    )

    # Model for prediction (just the y_pred output)
    prediction_model = tf.keras.models.Model(inputs=image_input, outputs=y_pred)

    # Compile with dummy loss (CTC loss is computed in the Lambda layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=5.0),
        loss={'ctc_loss': lambda y_true, y_pred: y_pred}
    )

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
