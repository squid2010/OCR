# predict.py
"""
Prediction module for Medical Handwriting OCR System.

Provides:
- predict_medicine_name: Python API for single image prediction
- CLI batch/single prediction logic (used by run_ocr.py)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image
import pickle

# --- CONFIGURATION ---

MODEL_PATH = os.path.join("models", "ocr_model_prediction.h5")
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

def get_confidence(pred):
    """
    Returns a confidence score for the prediction.
    """
    # Use mean max probability across timesteps as confidence
    return float(np.mean(np.max(pred, axis=-1)))

# --- MAIN PREDICTION API ---

def predict_medicine_name(image_path, model_path=MODEL_PATH, char_map_path=CHAR_MAP_PATH):
    """
    Predicts the medicine name from a single image.
    Returns:
        {
            "medicine_name": str,
            "confidence": float
        }
    """
    # Load model and mappings (cache if used in batch)
    model = load_model(model_path, compile=False)
    char_to_num, num_to_char = load_char_mappings(char_map_path)

    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # (1, H, W, 1)

    preds = model.predict(img)
    pred = preds[0]  # (timesteps, num_classes)

    # Decode prediction
    input_len = np.ones((1,)) * pred.shape[0]
    decoded, _ = K.ctc_decode(np.expand_dims(pred, axis=0), input_length=input_len, greedy=True)
    out = K.get_value(decoded[0])[0]
    medicine_name = ''.join([num_to_char.get(i, '') for i in out if i != -1])

    confidence = get_confidence(pred)
    return {
        "medicine_name": medicine_name,
        "confidence": confidence
    }

# --- BATCH PREDICTION ---

def batch_predict(folder_path, output_csv="results.csv", model_path=MODEL_PATH, char_map_path=CHAR_MAP_PATH):
    """
    Predicts medicine names for all images in a folder.
    Writes results to output_csv.
    """
    import csv

    model = load_model(model_path, compile=False)
    char_to_num, num_to_char = load_char_mappings(char_map_path)

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    results = []

    for fname in image_files:
        img_path = os.path.join(folder_path, fname)
        img = preprocess_image(img_path)
        img = np.expand_dims(img, axis=0)
        preds = model.predict(img)
        pred = preds[0]
        input_len = np.ones((1,)) * pred.shape[0]
        decoded, _ = K.ctc_decode(np.expand_dims(pred, axis=0), input_length=input_len, greedy=True)
        out = K.get_value(decoded[0])[0]
        medicine_name = ''.join([num_to_char.get(i, '') for i in out if i != -1])
        confidence = get_confidence(pred)
        results.append({
            "IMAGE": fname,
            "MEDICINE_NAME": medicine_name,
            "CONFIDENCE": confidence
        })

    # Write to CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["IMAGE", "MEDICINE_NAME", "CONFIDENCE"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Batch prediction complete. Results saved to {output_csv}")

# --- CLI ENTRY POINT (for run_ocr.py) ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Medical Handwriting OCR Prediction")
    parser.add_argument("--image", type=str, help="Path to single image for prediction")
    parser.add_argument("--folder", type=str, help="Path to folder for batch prediction")
    parser.add_argument("--output", type=str, default="results.csv", help="Output CSV for batch prediction")
    parser.add_argument("--visualize", action="store_true", help="Visualize prediction (single image only)")

    args = parser.parse_args()

    if args.image:
        result = predict_medicine_name(args.image)
        print(f"Predicted Medicine Name: {result['medicine_name']}")
        print(f"Confidence: {result['confidence']:.4f}")
        if args.visualize:
            import matplotlib.pyplot as plt
            img = Image.open(args.image)
            plt.imshow(img, cmap="gray")
            plt.title(f"Prediction: {result['medicine_name']} (Conf: {result['confidence']:.2f})")
            plt.axis("off")
            plt.show()
    elif args.folder:
        batch_predict(args.folder, args.output)
    else:
        print("Please provide --image or --folder argument for prediction.")
