#!/usr/bin/env python3
# OCR/check_model_loading.py
# Script to verify model architecture and weights compatibility before retraining

import os
import sys

def main():
    print("="*60)
    print("OCR Model Loading Compatibility Check")
    print("="*60)
    model_path = "models/ocr_model_prediction.keras"
    char_map_path = "models/char_mappings.pkl"

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        print("Please retrain your model and save it as a full model (.keras).")
        return
    if not os.path.exists(char_map_path):
        print(f"✗ Character mappings file not found: {char_map_path}")
        print("Please ensure you have a valid char_mappings.pkl file.")
        return

    print(f"✓ Found model file: {model_path}")
    print(f"✓ Found character mappings: {char_map_path}")

    # Try to load model
    try:
        import tensorflow as tf
        print("Attempting to load model...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✓ Model loaded successfully!")
        print("Model summary:")
        model.summary()
    except Exception as e:
        print("✗ Failed to load model!")
        print("Error details:")
        print(e)
        print("\nThis usually means the model architecture has changed since training.")
        print("To fix:")
        print("  1. Ensure your config.py and build_ocr_model() are identical for training and evaluation.")
        print("  2. Retrain your model and save as a full model (.keras).")
        print("  3. Delete any old model files in 'models/' before retraining.")
        return

    print("\nIf the summary above matches your expected architecture, you can proceed to evaluation.")
    print("If you see errors, retrain your model with the correct architecture.")

if __name__ == "__main__":
    main()
