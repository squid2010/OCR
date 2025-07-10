import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class OCRModel:
    def __init__(self, img_height=128, img_width=384, max_length=32, num_classes=None):
        self.img_height = img_height
        self.img_width = img_width
        self.max_length = max_length
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        input_img = layers.Input(shape=(self.img_height, self.img_width, 1), name='image')
        x = input_img

        # CNN feature extractor
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 1))(x)  # Keep width dimension larger

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 1))(x)

        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 1))(x)

        # Dropout for regularization
        x = layers.Dropout(0.2)(x)

        # Reshape for RNN
        # After pooling operations, height should be reduced significantly
        # x shape: (batch, 4, 96, 256)
        x = layers.Permute((2, 1, 3))(x)  # (batch, 96, 4, 256)
        x = layers.Reshape((96, 4 * 256))(x)  # (batch, 96, 1024)
        x = layers.Dense(256, activation='relu')(x)

        # Bidirectional LSTM layers
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(x)
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(x)

        # Output layer for CTC
        # +1 for CTC blank token
        outputs = layers.Dense(self.num_classes + 1, activation='softmax', name='dense_output')(x)

        self.model = Model(inputs=input_img, outputs=outputs, name='OCR_Model')
        return self.model

    def get_model(self):
        """Return the built model"""
        if self.model is None:
            self.build_model()
        return self.model

    def compile_model(self, learning_rate=0.001):
        """Compile the model with CTC loss"""
        if self.model is None:
            self.build_model()

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer)
        return self.model

class CTCLayer(layers.Layer):
    """CTC layer for computing CTC loss"""
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        # Compute the training-time loss value
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # Return the predictions for inference
        return y_pred

def build_training_model(img_height=128, img_width=384, max_length=32, num_classes=None):
    """Build complete training model with CTC loss"""

    # Input layers
    input_img = layers.Input(shape=(img_height, img_width, 1), name='image')
    labels = layers.Input(name='label', shape=(max_length,), dtype='int32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')

    # Build OCR model
    ocr_model = OCRModel(img_height, img_width, max_length, num_classes)
    predictions = ocr_model.build_model()(input_img)

    # CTC layer
    output = CTCLayer(name='ctc_loss')(labels, predictions, input_length, label_length)

    # Create training model
    training_model = Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=output,
        name='OCR_Training_Model'
    )

    return training_model, ocr_model

def decode_predictions(predictions, char_to_num_map, num_to_char_map):
    """Decode CTC predictions to text"""
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]

    # Use greedy search
    results = tf.keras.backend.ctc_decode(predictions, input_length=input_len, greedy=True)[0][0]

    # Convert to text
    output_text = []
    for res in results.numpy():
        text = ''.join([num_to_char_map.get(idx, '') for idx in res if idx != -1])
        # Remove consecutive duplicates and blanks
        text = ''.join([char for i, char in enumerate(text) if i == 0 or char != text[i-1]])
        text = text.replace('', '')  # Remove blank tokens if any
        output_text.append(text.strip())

    return output_text

def calculate_sequence_lengths(predictions):
    """Calculate sequence lengths for CTC"""
    # For CTC, input length is the width of the feature map after CNN
    batch_size = tf.shape(predictions)[0]
    input_length = tf.fill([batch_size], tf.shape(predictions)[1])
    return input_length

# Model configuration
# Removed get_model_config; use OCRConfig from config.py instead
