import os

# Suppress TensorFlow Metal/NUMA warnings for Apple Silicon
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data_loader import load_csv_labels, get_char_maps
from ocr_model import build_training_model, OCRModel, calculate_sequence_lengths
from config import OCRConfig
import pandas as pd

# --- DO NOT enable mixed precision for CTC on M1/M2 ---
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Print GPU info for Apple Silicon
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Using GPU:", gpus[0])
else:
    print("No GPU found. Training will be slow on CPU.")

class OCRTrainer:
    def __init__(self, config=None):
        # Use OCRConfig for all configuration
        if config is not None:
            self.config = config
        else:
            self.config = {
                'img_height': OCRConfig.IMG_HEIGHT,
                'img_width': OCRConfig.IMG_WIDTH,
                'img_channels': OCRConfig.IMG_CHANNELS,
                'max_length': OCRConfig.MAX_TEXT_LENGTH,
                'batch_size': OCRConfig.BATCH_SIZE,
                'epochs': OCRConfig.EPOCHS,
                'learning_rate': OCRConfig.LEARNING_RATE,
                'cnn_filters': OCRConfig.CNN_FILTERS,
                'lstm_units': OCRConfig.LSTM_UNITS,
                'dropout_rate': OCRConfig.DROPOUT_RATE,
                'use_bidirectional': OCRConfig.USE_BIDIRECTIONAL,
                'use_batch_norm': OCRConfig.USE_BATCH_NORMALIZATION,
            }
        self.char_to_num = None
        self.num_to_char = None
        self.training_model = None
        self.prediction_model = None
        self.history = None

    def prepare_data(self, train_csv_path, train_img_folder,
                     val_csv_path, val_img_folder,
                     test_csv_path, test_img_folder):
        """Prepare training, validation and test datasets"""

        print("Loading training data...")
        train_images, train_medicine_names, train_generic_names = load_csv_labels(train_csv_path, train_img_folder)

        print("Loading validation data...")
        val_images, val_medicine_names, val_generic_names = load_csv_labels(val_csv_path, val_img_folder)

        print("Loading test data...")
        test_images, test_medicine_names, test_generic_names = load_csv_labels(test_csv_path, test_img_folder)

        # Check if we have any training data
        if len(train_images) == 0:
            raise ValueError(f"No training images found in {train_img_folder}. Please check that the images referenced in {train_csv_path} exist.")

        # Combine all labels to build vocabulary
        all_labels = train_medicine_names + train_generic_names + \
                    val_medicine_names + val_generic_names + \
                    test_medicine_names + test_generic_names

        # Make sure we have some labels
        if not all_labels:
            raise ValueError("No text labels found in the datasets. Please check your CSV files.")

        # Create character mappings
        self.char_to_num, self.num_to_char = get_char_maps(all_labels)
        print(f"Vocabulary size: {len(self.char_to_num)}")
        print(f"Characters: {sorted(self.char_to_num.keys())}")

        # Validate vocabulary size
        if len(self.char_to_num) <= 1:
            raise ValueError("Vocabulary too small. Found only empty or identical text in labels.")

        # Update model config with vocabulary size
        self.config['num_classes'] = len(self.char_to_num)

        # For this implementation, we'll focus on medicine names
        self.train_labels = train_medicine_names
        self.val_labels = val_medicine_names
        self.test_labels = test_medicine_names

        self.train_images = train_images
        self.val_images = val_images
        self.test_images = test_images

        print(f"Training samples: {len(self.train_images)}")
        print(f"Validation samples: {len(self.val_images)}")
        print(f"Test samples: {len(self.test_images)}")

        if len(self.train_images) == 0:
            raise ValueError("No training images found. Please check the path to the training data.")

    def create_ctc_dataset(self, image_paths, labels, is_training=True):
        """Create dataset with CTC-compatible format for model.fit()"""

        # Pre-encode all labels to avoid tf.py_function issues
        encoded_labels = []
        label_lengths = []

        for label in labels:
            label_str = str(label)
            label_str = label_str[:self.config['max_length']]
            encoded = []
            for char in label_str:
                if char in self.char_to_num:
                    encoded.append(self.char_to_num[char])
                else:
                    encoded.append(0)  # Unknown character

            # Pad to max_length with -1 (CTC expects -1 for padding)
            while len(encoded) < self.config['max_length']:
                encoded.append(-1)
            encoded_labels.append(encoded)
            label_lengths.append(min(len(label_str), self.config['max_length']))

        def preprocess_data(image_path, label_encoded, label_length):
            img = tf.io.read_file(tf.cast(image_path, tf.string))
            img = tf.image.decode_png(img, channels=1)
            img = tf.image.resize(img, [self.config['img_height'], self.config['img_width']])
            img = tf.cast(img, tf.float32) / 255.0

            # For CTC, input_length is the time steps after CNN (here, 96)
            input_length = tf.constant(96, dtype=tf.int32)
            inputs = {
                'image': img,
                'label': label_encoded,
                'input_length': input_length,
                'label_length': label_length
            }
            # Dummy target (not used, but required by .fit())
            targets = tf.zeros([1])
            return (inputs, targets)

        dataset = tf.data.Dataset.from_tensor_slices((
            tf.constant(image_paths, dtype=tf.string),
            tf.constant(encoded_labels, dtype=tf.int32),
            tf.constant(label_lengths, dtype=tf.int32)
        ))
        dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(OCRConfig.BATCH_SIZE)
        dataset = dataset.prefetch(1)
        return dataset

    def build_model(self):
        """Build the training and prediction models"""
        print("Building OCR model...")

        self.training_model, self.ocr_model = build_training_model(
            img_height=self.config['img_height'],
            img_width=self.config['img_width'],
            max_length=self.config['max_length'],
            num_classes=self.config['num_classes']
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        self.training_model.compile(
            optimizer=optimizer,
            run_eagerly=False
        )

        self.prediction_model = self.ocr_model.get_model()
        print("Model built successfully!")

    def train(self, train_dataset, val_dataset, model_save_path='models/ocr_model'):
        """Train the OCR model using model.fit() and Keras callbacks. Always save best weights as final model if training fails."""
        import os
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=OCRConfig.REDUCE_LR_FACTOR,
                patience=OCRConfig.REDUCE_LR_PATIENCE,
                min_lr=OCRConfig.MIN_LEARNING_RATE,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=OCRConfig.EARLY_STOPPING_PATIENCE,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_save_path + '_best.weights.h5',
                monitor='loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        ]

        print(f"Starting training for {OCRConfig.EPOCHS} epochs...")

        try:
            self.history = self.training_model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=OCRConfig.EPOCHS,
                callbacks=callbacks
            )
            # Save final weights after successful training
            self.training_model.save_weights(model_save_path + '_final.weights.h5')
        except Exception as e:
            print(f"Exception during training: {e}")
            import traceback
            traceback.print_exc()
            print("Training stopped due to an error.")
            # Try to load best weights if available
            best_weights_path = model_save_path + '_best.weights.h5'
            if os.path.exists(best_weights_path):
                print("Loading best weights after failure...")
                self.training_model.load_weights(best_weights_path)
            else:
                print("No best weights found to load after failure.")

        # Always save the prediction model after training or failure
        self.prediction_model.save(model_save_path + '_prediction.h5')

        # Plot and save the loss curves if available
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            if hasattr(self, 'history') and self.history is not None:
                plt.plot(self.history.history['loss'], label='Train Loss')
                if 'val_loss' in self.history.history:
                    plt.plot(self.history.history['val_loss'], label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.savefig('training_history.png')
        except Exception as e:
            print(f"Exception during plotting: {e}")

    def train(self, train_dataset, val_dataset, model_save_path='models/ocr_model'):
        """Train the OCR model using model.fit() and Keras callbacks"""
        import os
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=OCRConfig.REDUCE_LR_FACTOR,
                patience=OCRConfig.REDUCE_LR_PATIENCE,
                min_lr=OCRConfig.MIN_LEARNING_RATE,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=OCRConfig.EARLY_STOPPING_PATIENCE,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_save_path + '_best.weights.h5',
                monitor='loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        ]

        print(f"Starting training for {OCRConfig.EPOCHS} epochs...")

        self.history = self.training_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=OCRConfig.EPOCHS,
            callbacks=callbacks
        )

        # Save final weights and model
        self.training_model.save_weights(model_save_path + '_final.weights.h5')
        self.prediction_model.save(model_save_path + '_prediction.h5')

        # Plot and save the loss curves
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(self.history.history['loss'], label='Train Loss')
            if 'val_loss' in self.history.history:
                plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.savefig('training_history.png')
        except Exception as e:
            print(f"Exception during plotting: {e}")


def main():
    """Main training function"""

    # Configuration
    config = {
        'img_height': OCRConfig.IMG_HEIGHT,
        'img_width': OCRConfig.IMG_WIDTH,
        'img_channels': OCRConfig.IMG_CHANNELS,
        'max_length': OCRConfig.MAX_TEXT_LENGTH,
        'batch_size': OCRConfig.BATCH_SIZE,
        'epochs': OCRConfig.EPOCHS,
        'learning_rate': OCRConfig.LEARNING_RATE,
        'cnn_filters': OCRConfig.CNN_FILTERS,
        'lstm_units': OCRConfig.LSTM_UNITS,
        'dropout_rate': OCRConfig.DROPOUT_RATE,
        'use_bidirectional': OCRConfig.USE_BIDIRECTIONAL,
        'use_batch_norm': OCRConfig.USE_BATCH_NORMALIZATION,
    }

    # Paths
    base_path = 'trainingData'
    train_csv = os.path.join(base_path, 'Training', 'training_labels.csv')
    train_folder = os.path.join(base_path, 'Training')
    train_img_folder = os.path.join(train_folder, 'training_words')

    val_csv = os.path.join(base_path, 'Validation', 'validation_labels.csv')
    val_folder = os.path.join(base_path, 'Validation')
    val_img_folder = os.path.join(val_folder, 'validation_words')

    test_csv = os.path.join(base_path, 'Testing', 'testing_labels.csv')
    test_folder = os.path.join(base_path, 'Testing')
    test_img_folder = os.path.join(test_folder, 'testing_words')

    # Check if the dataset directories exist
    for path in [train_folder, val_folder, test_folder]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
        else:
            print(f"Directory exists: {path}")

    # Check if image directories exist, create them if needed
    for path in [train_img_folder, val_img_folder, test_img_folder]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created image directory: {path}")
        else:
            print(f"Image directory exists: {path}")

    # Check if the CSV files exist
    missing_files = []
    for csv_file in [train_csv, val_csv, test_csv]:
        if not os.path.exists(csv_file):
            print(f"Warning: CSV file not found: {csv_file}")
            missing_files.append(csv_file)

    if train_csv in missing_files:
        print("\nERROR: Training data CSV file is missing!")
        print("Please create the training data first by running:")
        print("  python run_ocr.py --setup-data")
        return

    # Initialize trainer
    trainer = OCRTrainer(config)

    # Prepare data
    try:
        # Make sure image folders exist - they should be the word directories
        # Check first for the expected subdirectories, otherwise use the directory itself
        if os.path.exists(train_img_folder):
            print(f"Using training image folder: {train_img_folder}")
        elif os.path.exists(train_folder):
            print(f"Training image subdirectory not found, using: {train_folder}")
            train_img_folder = train_folder

        if os.path.exists(val_img_folder):
            print(f"Using validation image folder: {val_img_folder}")
        elif os.path.exists(val_folder):
            print(f"Validation image subdirectory not found, using: {val_folder}")
            val_img_folder = val_folder

        if os.path.exists(test_img_folder):
            print(f"Using testing image folder: {test_img_folder}")
        elif os.path.exists(test_folder):
            print(f"Testing image subdirectory not found, using: {test_folder}")
            test_img_folder = test_folder

        trainer.prepare_data(
            train_csv, train_img_folder,
            val_csv, val_img_folder,
            test_csv, test_img_folder
        )
    except Exception as e:
        print(f"Error preparing data: {e}")
        return

    # Create datasets
    print("Creating datasets...")
    try:
        train_dataset = trainer.create_ctc_dataset(trainer.train_images, trainer.train_labels, is_training=True)
        val_dataset = trainer.create_ctc_dataset(trainer.val_images, trainer.val_labels, is_training=False)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease make sure the training data exists at the specified path:")
        print(f"  Training data: {train_csv}")
        print(f"  Training folder: {train_folder}")
        print("\nCheck that the image files referenced in the CSV file exist in the folder.")
        return

    # Build model
    trainer.build_model()

    # Print model summary
    print("\nModel Summary:")
    trainer.training_model.summary()

    # Debug: Print model output shape for CTC time steps
    dummy_input = tf.zeros([1, config['img_height'], config['img_width'], 1])
    cnn_output = trainer.prediction_model(dummy_input)
    print("Model output shape (for CTC time steps):", cnn_output.shape)

    # Train model
    trainer.train(train_dataset, val_dataset)

    # Save character mappings
    import pickle
    with open('models/char_mappings.pkl', 'wb') as f:
        pickle.dump({
            'char_to_num': trainer.char_to_num,
            'num_to_char': trainer.num_to_char
        }, f)

    print("Training completed!")
    print("Model saved to 'models/' directory")
    print("Character mappings saved to 'models/char_mappings.pkl'")

if __name__ == "__main__":
    main()
