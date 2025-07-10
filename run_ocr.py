#!/usr/bin/env python3
"""
Utility script for easy OCR model training and testing
This script provides a simple interface to train, evaluate, and test the OCR model
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import tensorflow as tf
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        print(f"✓ TensorFlow {tf.__version__} is installed")
        print(f"✓ GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False
    return True

def check_data_structure():
    """Check if data is properly structured"""
    base_path = Path("trainingData")

    required_files = [
        "Training/training_labels.csv",
        "Validation/validation_labels.csv",
        "Testing/testing_labels.csv"
    ]

    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(str(full_path))

    if missing_files:
        print("✗ Missing required data files:")
        for file in missing_files:
            print(f"  - {file}")
        return False

    print("✓ All required data files found")
    return True

def train_model():
    """Train the OCR model"""
    print("Starting model training...")
    print("=" * 50)

    if not check_dependencies():
        return False

    if not check_data_structure():
        return False

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Run training
    try:
        result = subprocess.run([sys.executable, "train.py"],
                              capture_output=False, text=True)
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("✓ Training completed successfully!")
            print("Check the 'models' directory for saved model files")
            return True
        else:
            print("\n" + "=" * 50)
            print("✗ Training failed!")
            return False
    except Exception as e:
        print(f"Error during training: {e}")
        return False

def evaluate_model():
    """Evaluate the trained model"""
    print("Starting model evaluation...")
    print("=" * 50)

    # Check if model exists
    model_path = "models/ocr_model_prediction.h5"
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("Please train the model first using: python run_ocr.py --train")
        return False

    # Run evaluation
    try:
        result = subprocess.run([sys.executable, "evaluate.py"],
                              capture_output=False, text=True)
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("✓ Evaluation completed successfully!")
            print("Check the evaluation reports and plots generated")
            return True
        else:
            print("\n" + "=" * 50)
            print("✗ Evaluation failed!")
            return False
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return False

def predict_image(image_path, visualize=False):
    """Predict text from a single image"""
    print(f"Predicting text from: {image_path}")
    print("=" * 50)

    # Check if model exists
    model_path = "models/ocr_model_prediction.h5"
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("Please train the model first using: python run_ocr.py --train")
        return False

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        return False

    # Run prediction
    try:
        cmd = [sys.executable, "predict.py", "--image", image_path]
        if visualize:
            cmd.append("--visualize")

        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("✓ Prediction completed successfully!")
            return True
        else:
            print("\n" + "=" * 50)
            print("✗ Prediction failed!")
            return False
    except Exception as e:
        print(f"Error during prediction: {e}")
        return False

def predict_folder(folder_path, output_file=None):
    """Predict text from all images in a folder"""
    print(f"Predicting text from folder: {folder_path}")
    print("=" * 50)

    # Check if model exists
    model_path = "models/ocr_model_prediction.h5"
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("Please train the model first using: python run_ocr.py --train")
        return False

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"✗ Folder not found: {folder_path}")
        return False

    # Run prediction
    try:
        cmd = [sys.executable, "predict.py", "--folder", folder_path]
        if output_file:
            cmd.extend(["--output", output_file])

        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("✓ Batch prediction completed successfully!")
            if output_file:
                print(f"Results saved to: {output_file}")
            return True
        else:
            print("\n" + "=" * 50)
            print("✗ Batch prediction failed!")
            return False
    except Exception as e:
        print(f"Error during prediction: {e}")
        return False

def show_status():
    """Show current status of the OCR system"""
    print("OCR System Status")
    print("=" * 50)

    # Check dependencies
    print("Dependencies:")
    if check_dependencies():
        print("✓ All dependencies installed")
    else:
        print("✗ Missing dependencies")

    print()

    # Check data
    print("Data:")
    if check_data_structure():
        print("✓ Data structure is correct")
    else:
        print("✗ Data structure issues found")

    print()

    # Check models
    print("Models:")
    model_files = [
        "models/ocr_model_prediction.h5",
        "models/char_mappings.pkl"
    ]

    all_models_exist = True
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✓ {model_file}")
        else:
            print(f"✗ {model_file}")
            all_models_exist = False

    if all_models_exist:
        print("✓ All required model files found")
    else:
        print("✗ Some model files missing - run training first")

def main():
    parser = argparse.ArgumentParser(
        description="OCR Model Training and Testing Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python run_ocr.py --train

  # Evaluate the model
  python run_ocr.py --evaluate

  # Predict single image
  python run_ocr.py --predict image.png --visualize

  # Predict all images in folder
  python run_ocr.py --predict-folder images/ --output results.csv

  # Check system status
  python run_ocr.py --status
        """
    )

    parser.add_argument("--train", action="store_true",
                       help="Train the OCR model")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate the trained model")
    parser.add_argument("--predict", type=str, metavar="IMAGE_PATH",
                       help="Predict text from a single image")
    parser.add_argument("--predict-folder", type=str, metavar="FOLDER_PATH",
                       help="Predict text from all images in folder")
    parser.add_argument("--output", type=str, metavar="OUTPUT_FILE",
                       help="Output file for batch predictions (CSV format)")
    parser.add_argument("--visualize", action="store_true",
                       help="Show visualization for single image prediction")
    parser.add_argument("--status", action="store_true",
                       help="Show system status")

    args = parser.parse_args()

    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return

    # Execute requested actions
    if args.status:
        show_status()

    if args.train:
        success = train_model()
        if not success:
            sys.exit(1)

    if args.evaluate:
        success = evaluate_model()
        if not success:
            sys.exit(1)

    if args.predict:
        success = predict_image(args.predict, args.visualize)
        if not success:
            sys.exit(1)

    if args.predict_folder:
        success = predict_folder(args.predict_folder, args.output)
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()
