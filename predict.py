import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import pickle
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from ocr_model import OCRModel
from config import OCRConfig

class OCRPredictor:
    def __init__(self, model_path=None, char_mappings_path=None, img_height=None, img_width=None):
        """Initialize OCR predictor with trained model and character mappings"""
        # Use OCRConfig for all config values if not provided
        self.model_path = model_path if model_path is not None else OCRConfig.PREDICTION_MODEL_PATH
        self.char_mappings_path = char_mappings_path if char_mappings_path is not None else OCRConfig.CHAR_MAPPINGS_PATH
        self.img_height = img_height if img_height is not None else OCRConfig.IMG_HEIGHT
        self.img_width = img_width if img_width is not None else OCRConfig.IMG_WIDTH
        self.model = None
        self.char_to_num = None
        self.num_to_char = None
        self.load_model_and_mappings()

    def load_model_and_mappings(self):
        """Load trained model and character mappings"""
        print("Loading model and character mappings...")

        # Load character mappings
        try:
            with open(self.char_mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                self.char_to_num = mappings['char_to_num']
                self.num_to_char = mappings['num_to_char']
            print(f"Loaded character mappings with {len(self.char_to_num)} characters")
        except FileNotFoundError:
            print(f"Character mappings file not found: {self.char_mappings_path}")
            sys.exit(1)

        # Load model
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        try:
            # Read image
            if isinstance(image_path, str):
                # Check if the path exists, if not try with words subdirectory
                if not os.path.exists(image_path):
                    # Try with words subfolder
                    dirname = os.path.dirname(image_path)
                    basename = os.path.basename(dirname)
                    new_path = os.path.join(dirname, f"{basename.lower()}_words", os.path.basename(image_path))
                    if os.path.exists(new_path):
                        image_path = new_path
                        print(f"Using image path: {image_path}")

                img = tf.io.read_file(image_path)
                img = tf.image.decode_image(img, channels=1)
            else:
                # If image_path is already a numpy array
                img = tf.convert_to_tensor(image_path)
                if len(img.shape) == 3 and img.shape[-1] == 3:
                    img = tf.image.rgb_to_grayscale(img)
                elif len(img.shape) == 2:
                    img = tf.expand_dims(img, -1)

            # Resize to model input size
            img = tf.image.resize(img, [self.img_height, self.img_width])

            # Normalize to [0, 1]
            img = tf.cast(img, tf.float32) / 255.0

            # Add batch dimension
            img = tf.expand_dims(img, 0)

            return img

        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def decode_predictions(self, predictions):
        """Decode model predictions to text"""
        input_len = np.ones(predictions.shape[0]) * predictions.shape[1]

        # Use CTC greedy decoder
        try:
            results = tf.keras.backend.ctc_decode(
                predictions,
                input_length=input_len,
                greedy=True
            )[0][0]

            # Convert to text
            output_text = []
            for res in results.numpy():
                text = ''.join([self.num_to_char.get(idx, '') for idx in res if idx >= 0])
                # Remove consecutive duplicates
                if text:
                    cleaned_text = text[0] if text else ""
                    for i in range(1, len(text)):
                        if text[i] != text[i-1]:
                            cleaned_text += text[i]
                    output_text.append(cleaned_text.strip())
                else:
                    output_text.append('')

            return output_text
        except Exception as e:
            print(f"Error decoding predictions: {e}")
            return [""]

    def predict_text(self, image_path, confidence_threshold=0.5):
        """Predict text from image with confidence scoring"""
        # Preprocess image
        img = self.preprocess_image(image_path)
        if img is None:
            return "", 0.0

        try:
            # Get predictions
            predictions = self.model(img)

            # Calculate confidence (average of max probabilities)
            max_probs = tf.reduce_max(predictions, axis=-1)
            confidence = tf.reduce_mean(max_probs).numpy()

            # Decode predictions
            decoded_text = self.decode_predictions(predictions)
            predicted_text = decoded_text[0] if decoded_text else ""

            return predicted_text, float(confidence)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return "", 0.0

    def predict_batch(self, image_paths, batch_size=16):
        """Predict text for multiple images"""
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []

            # Preprocess batch
            for path in batch_paths:
                img = self.preprocess_image(path)
                if img is not None:
                    batch_images.append(img[0])  # Remove batch dimension for concatenation
                else:
                    # Add placeholder for failed images
                    batch_images.append(tf.zeros((self.img_height, self.img_width, 1)))

            if batch_images:
                # Stack into batch
                batch_tensor = tf.stack(batch_images)

                try:
                    # Get predictions
                    predictions = self.model(batch_tensor)

                    # Calculate confidences
                    max_probs = tf.reduce_max(predictions, axis=-1)
                    confidences = tf.reduce_mean(max_probs, axis=-1).numpy()

                    # Decode predictions
                    decoded_texts = self.decode_predictions(predictions)

                    # Store results
                    for j, (path, text, conf) in enumerate(zip(batch_paths, decoded_texts, confidences)):
                        results.append({
                            'image_path': path,
                            'predicted_text': text,
                            'confidence': float(conf)
                        })

                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # Add error results
                    for path in batch_paths:
                        results.append({
                            'image_path': path,
                            'predicted_text': "",
                            'confidence': 0.0
                        })

        return results

    def visualize_prediction(self, image_path, save_path=None):
        """Visualize image with prediction"""
        # Make prediction
        predicted_text, confidence = self.predict_text(image_path)

        # Load and display image
        try:
            if isinstance(image_path, str):
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = image_path

            plt.figure(figsize=(12, 4))
            plt.imshow(img, cmap='gray')
            plt.title(f'Predicted: "{predicted_text}" (Confidence: {confidence:.3f})')
            plt.axis('off')

            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                print(f"Visualization saved to: {save_path}")

            plt.show()

        except Exception as e:
            print(f"Error visualizing image: {e}")

        return predicted_text, confidence

def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='OCR Prediction Tool')
    parser.add_argument('--model', type=str, default=OCRConfig.PREDICTION_MODEL_PATH,
                       help='Path to trained model')
    parser.add_argument('--mappings', type=str, default=OCRConfig.CHAR_MAPPINGS_PATH,
                       help='Path to character mappings file')
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
    parser.add_argument('--folder', type=str, help='Path to folder containing images')
    parser.add_argument('--output', type=str, help='Path to save results CSV')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--batch-size', type=int, default=OCRConfig.BATCH_SIZE, help='Batch size for processing')

    args = parser.parse_args()

    # Check if model files exist
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Please train the model first using train.py")
        return

    if not os.path.exists(args.mappings):
        print(f"Character mappings file not found: {args.mappings}")
        print("Please train the model first using train.py")
        return

    # Initialize predictor
    predictor = OCRPredictor(args.model, args.mappings, img_height=OCRConfig.IMG_HEIGHT, img_width=OCRConfig.IMG_WIDTH)

    # Single image prediction
    if args.image:
        image_path = args.image
        # Check if the path exists, if not try with words subdirectory
        if not os.path.exists(image_path):
            # Try with words subfolder
            dirname = os.path.dirname(image_path)
            basename = os.path.basename(dirname)
            new_path = os.path.join(dirname, f"{basename.lower()}_words", os.path.basename(image_path))
            if os.path.exists(new_path):
                image_path = new_path
                print(f"Using image path: {image_path}")

        if os.path.exists(image_path):
            print(f"Predicting text for: {image_path}")

            if args.visualize:
                predicted_text, confidence = predictor.visualize_prediction(image_path)
            else:
                predicted_text, confidence = predictor.predict_text(image_path)

            print(f"Predicted text: '{predicted_text}'")
            print(f"Confidence: {confidence:.3f}")

        else:
            print(f"Image file not found: {args.image}")

    # Batch prediction on folder
    elif args.folder:
        if os.path.exists(args.folder):
            print(f"Processing images in folder: {args.folder}")

            # Check if there's a words subfolder
            words_subfolder = os.path.join(args.folder, f"{os.path.basename(args.folder).lower()}_words")
            if os.path.exists(words_subfolder):
                folder_path = words_subfolder
                print(f"Using image subfolder: {folder_path}")
            else:
                folder_path = args.folder

            # Get all image files
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
            image_paths = []

            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(folder_path, file))

            if not image_paths:
                print("No image files found in the folder")
                return

            print(f"Found {len(image_paths)} image files")

            # Process batch
            results = predictor.predict_batch(image_paths, args.batch_size)

            # Display results
            print("\nPrediction Results:")
            print("-" * 80)
            for result in results:
                print(f"File: {os.path.basename(result['image_path'])}")
                print(f"Text: '{result['predicted_text']}'")
                print(f"Confidence: {result['confidence']:.3f}")
                print("-" * 40)

            # Save results if output path provided
            if args.output:
                import pandas as pd
                df = pd.DataFrame(results)
                df.to_csv(args.output, index=False)
                print(f"\nResults saved to: {args.output}")

        else:
            print(f"Folder not found: {args.folder}")

    else:
        print("Please provide either --image or --folder argument")
        parser.print_help()

def predict_from_array(image_array, model_path=None, char_mappings_path=None):
    """Utility function to predict from numpy array"""
    predictor = OCRPredictor(
        model_path=model_path if model_path is not None else OCRConfig.PREDICTION_MODEL_PATH,
        char_mappings_path=char_mappings_path if char_mappings_path is not None else OCRConfig.CHAR_MAPPINGS_PATH,
        img_height=OCRConfig.IMG_HEIGHT,
        img_width=OCRConfig.IMG_WIDTH
    )
    return predictor.predict_text(image_array)

def predict_medicine_name(image_path, model_path=None, char_mappings_path=None):
    """Convenience function for medicine name prediction"""
    predictor = OCRPredictor(
        model_path=model_path if model_path is not None else OCRConfig.PREDICTION_MODEL_PATH,
        char_mappings_path=char_mappings_path if char_mappings_path is not None else OCRConfig.CHAR_MAPPINGS_PATH,
        img_height=OCRConfig.IMG_HEIGHT,
        img_width=OCRConfig.IMG_WIDTH
    )
    text, confidence = predictor.predict_text(image_path)

    return {
        'medicine_name': text,
        'confidence': confidence,
        'success': confidence > OCRConfig.CONFIDENCE_THRESHOLD
    }

if __name__ == "__main__":
    main()
