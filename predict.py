# predict.py
"""
Prediction module for Medical Handwriting OCR System.

Provides:
- predict_medicine_name: Python API for single image prediction
- CLI batch/single prediction logic (used by run_ocr.py)
- Snap predictions to closest valid medicine name using edit distance
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image
import pickle
import pandas as pd

# --- CONFIGURATION ---

MODEL_PATH = os.path.join("models", "ocr_model_final.keras")
CHAR_MAP_PATH = os.path.join("models", "char_mappings.pkl")

IMG_HEIGHT = 128
IMG_WIDTH = 384
MAX_TEXT_LENGTH = 32

# --- UTILS ---

def load_char_mappings(char_map_path=CHAR_MAP_PATH):
    with open(char_map_path, "rb") as f:
        mappings = pickle.load(f)
    char_to_num = mappings["char_to_num"]
    num_to_char = mappings["num_to_char"]
    return char_to_num, num_to_char

def preprocess_image(image_path):
    """
    Loads and preprocesses an image for OCR model.
    - Converts to grayscale
    - Resizes to (IMG_HEIGHT, IMG_WIDTH)
    - Normalizes to [0, 1]
    """
    img = Image.open(image_path).convert("L")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # (H, W, 1)
    return img

def decode_prediction(pred, num_to_char):
    """
    Decodes a single CTC prediction to string.
    """
    # pred: (timesteps, num_classes)
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use best path decoding (greedy)
    decoded, _ = K.ctc_decode(np.expand_dims(pred, axis=0), input_length=np.expand_dims(pred.shape[0], axis=0), greedy=True)
    out = K.get_value(decoded[0])[0]
    text = ''.join([num_to_char.get(i, '') for i in out if i != -1])
    return text

def load_valid_medicine_names(csv_paths):
    """Load all unique medicine names from provided CSV files."""
    names = set()
    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            names.update(df['MEDICINE_NAME'].astype(str).unique())
    return list(names)

def find_closest_name(prediction, valid_names, max_distance=2):
    """Return the closest valid medicine name if within max_distance, else original prediction."""
    try:
        import Levenshtein
        distance_func = Levenshtein.distance
    except ImportError:
        # fallback to pure python implementation
        def distance_func(s1, s2):
            if len(s1) < len(s2):
                return distance_func(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row

def load_valid_medicine_names(csv_paths):
    """Load all unique medicine names from provided CSV files."""
    names = set()
    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            names.update(df['MEDICINE_NAME'].astype(str).unique())
    return list(names)

def find_closest_name(prediction, valid_names, max_distance=2):
    """Return the closest valid medicine name if within max_distance, else original prediction."""
    try:
        import Levenshtein
        distance_func = Levenshtein.distance
    except ImportError:
        # fallback to pure python implementation
        def distance_func(s1, s2):
            if len(s1) < len(s2):
                return distance_func(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
        distance_func = distance_func

    closest = prediction
    min_dist = max_distance + 1
    for name in valid_names:
        dist = distance_func(prediction, name)
        if dist < min_dist:
            min_dist = dist
            closest = name
    return closest if min_dist <= max_distance else prediction

def get_confidence(pred):
    """
    Returns a confidence score for the prediction.
    """
    # Use mean max probability across timesteps as confidence
    return float(np.mean(np.max(pred, axis=-1)))

# --- MAIN PREDICTION API ---

def predict_medicine_name(
    image,
    model_path=MODEL_PATH,
    char_map_path=CHAR_MAP_PATH,
    valid_names=None,
    max_distance=2,
    from_array=False
):
    """
    Predicts the medicine name from a single image (file path or numpy array).
    Args:
        image: str (path) or np.ndarray (H, W, 1) or (H, W)
        from_array: If True, treat image as numpy array
    Returns:
        {
            "medicine_name": snapped_name,
            "confidence": confidence,
            "raw_prediction": medicine_name
        }
    """
    model = load_model(model_path, compile=False)
    char_to_num, num_to_char = load_char_mappings(char_map_path)

    if from_array:
        img = image
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img = np.expand_dims(img, axis=0)
    else:
        img = preprocess_image(image)
        img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    pred = preds[0]

    input_len = np.ones((1,)) * pred.shape[0]
    decoded, _ = K.ctc_decode(np.expand_dims(pred, axis=0), input_length=input_len, greedy=True)
    out = K.get_value(decoded[0])[0]
    medicine_name = ''.join([num_to_char.get(i, '') for i in out if i != -1])

    confidence = get_confidence(pred)

    # Snap to closest valid name if needed
    if valid_names is not None:
        snapped_name = find_closest_name(medicine_name, valid_names, max_distance)
    else:
        snapped_name = medicine_name

    return {
        "medicine_name": snapped_name,
        "confidence": confidence,
        "raw_prediction": medicine_name
    }

# --- BATCH PREDICTION ---

def batch_predict(
    images,
    output_csv="results.csv",
    model_path=MODEL_PATH,
    char_map_path=CHAR_MAP_PATH,
    valid_names=None,
    max_distance=2,
    image_files=None,
    from_array=False
):
    """
    Predicts medicine names for all images in a folder or numpy array batch.
    Writes results to output_csv.
    Args:
        images: folder path (str) or numpy array (N, H, W, 1)
        image_files: list of filenames (optional, for numpy array batch)
        from_array: If True, treat images as numpy array batch
    """
    import csv

    model = load_model(model_path, compile=False)
    char_to_num, num_to_char = load_char_mappings(char_map_path)

    results = []

    if from_array:
        # images: numpy array (N, H, W, 1)
        imgs_np = images.astype(np.float32)
        if imgs_np.max() > 1.0:
            imgs_np = imgs_np / 255.0
        N = imgs_np.shape[0]
        preds = model.predict(imgs_np)
        input_lens = np.ones((N,)) * preds.shape[1]
        decoded, _ = K.ctc_decode(preds, input_length=input_lens, greedy=True)
        decoded_indices = K.get_value(decoded[0])
        for i, seq in enumerate(decoded_indices):
            medicine_name = ''.join([num_to_char.get(idx, '') for idx in seq if idx != -1])
            confidence = get_confidence(preds[i])
            if valid_names is not None:
                snapped_name = find_closest_name(medicine_name, valid_names, max_distance)
            else:
                snapped_name = medicine_name
            fname = image_files[i] if image_files is not None else f"img_{i}.png"
            results.append({
                "IMAGE": fname,
                "MEDICINE_NAME": snapped_name,
                "CONFIDENCE": confidence,
                "RAW_PREDICTION": medicine_name
            })
    else:
        # images: folder path (str)
        # Load all images in one step
        from data_loader import load_batch_images
        from config import OCRConfig
        imgs_np, image_files = load_batch_images(images, OCRConfig)
        N = imgs_np.shape[0]
        preds = model.predict(imgs_np)
        input_lens = np.ones((N,)) * preds.shape[1]
        decoded, _ = K.ctc_decode(preds, input_length=input_lens, greedy=True)
        decoded_indices = K.get_value(decoded[0])
        for i, seq in enumerate(decoded_indices):
            medicine_name = ''.join([num_to_char.get(idx, '') for idx in seq if idx != -1])
            confidence = get_confidence(preds[i])
            if valid_names is not None:
                snapped_name = find_closest_name(medicine_name, valid_names, max_distance)
            else:
                snapped_name = medicine_name
            fname = image_files[i] if image_files is not None else f"img_{i}.png"
            results.append({
                "IMAGE": fname,
                "MEDICINE_NAME": snapped_name,
                "CONFIDENCE": confidence,
                "RAW_PREDICTION": medicine_name
            })

    # Write results to CSV
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["IMAGE", "MEDICINE_NAME", "CONFIDENCE", "RAW_PREDICTION"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

# --- CLI ENTRY POINT (for run_ocr.py) ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Medical Handwriting OCR Prediction")
    parser.add_argument("--image", type=str, help="Path to single image for prediction")
    parser.add_argument("--folder", type=str, help="Path to folder for batch prediction")
    parser.add_argument("--output", type=str, default="results.csv", help="Output CSV for batch prediction")
    parser.add_argument("--visualize", action="store_true", help="Visualize prediction (single image only)")
    parser.add_argument("--np-image", action="store_true", help="Treat --image as numpy array file (.npy)")
    parser.add_argument("--np-batch", action="store_true", help="Treat --folder as numpy array batch file (.npy)")

    args = parser.parse_args()

    # Load valid medicine names for snapping
    valid_names = load_valid_medicine_names([
        "trainingData/Training/training_labels.csv",
        "trainingData/Validation/validation_labels.csv",
        "trainingData/Testing/testing_labels.csv"
    ])

    if args.image:
        if args.np_image:
            import numpy as np
            img_array = np.load(args.image)
            result = predict_medicine_name(img_array, valid_names=valid_names, from_array=True)
        else:
            result = predict_medicine_name(args.image, valid_names=valid_names)
        print(f"Predicted Medicine Name: {result['medicine_name']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Raw Prediction: {result['raw_prediction']}")
        if args.visualize:
            import matplotlib.pyplot as plt
            if args.np_image:
                img = img_array.squeeze()
                plt.imshow(img, cmap="gray")
            else:
                img = Image.open(args.image)
                plt.imshow(img, cmap="gray")
            plt.title(f"Prediction: {result['medicine_name']} (Conf: {result['confidence']:.2f})")
            plt.axis("off")
            plt.show()
    elif args.folder:
        if args.np_batch:
            import numpy as np
            images = np.load(args.folder)
            batch_predict(images, args.output, valid_names=valid_names, from_array=True)
        else:
            batch_predict(args.folder, args.output, valid_names=valid_names)
    else:
        print("Please provide --image or --folder argument for prediction.")
