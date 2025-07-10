#!/usr/bin/env python3
"""
Setup and Training Script for OCR Model

This script prepares sample data and trains the OCR model in one go.
It first creates sample training data and then runs the training process.
"""

import os
import sys
import subprocess
import argparse
import time

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

def check_python_dependencies():
    """Check if required dependencies are installed"""
    try:
        import tensorflow as tf
        import numpy as np
        import pandas as pd
        import PIL
        print(f"✓ TensorFlow {tf.__version__} is installed")
        print(f"✓ NumPy {np.__version__} is installed")
        print(f"✓ Pandas {pd.__version__} is installed")
        print(f"✓ PIL {PIL.__version__} is installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def create_sample_data(samples=100, clear=True):
    """Create sample data for training"""
    print_section_header("Creating Sample Data")

    cmd = [sys.executable, "prepare_sample_data.py",
           "--samples", str(samples),
           "--output", "trainingData"]

    if clear:
        cmd.append("--clear")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(result.stdout)
        print("✓ Sample data created successfully!")
        return True
    else:
        print("✗ Failed to create sample data!")
        print(result.stderr)
        return False

def train_model():
    """Train the OCR model"""
    print_section_header("Training OCR Model")

    result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True)

    if result.returncode == 0:
        print(result.stdout)
        print("✓ Model training completed successfully!")
        return True
    else:
        print("✗ Model training failed!")
        print(result.stderr)
        return False

def verify_data_structure():
    """Verify that the data structure is correct"""
    base_path = "trainingData"
    required_dirs = ["Training", "Validation", "Testing"]
    required_files = [
        os.path.join("Training", "training_labels.csv"),
        os.path.join("Validation", "validation_labels.csv"),
        os.path.join("Testing", "testing_labels.csv")
    ]

    # Check directories
    for directory in required_dirs:
        dir_path = os.path.join(base_path, directory)
        if not os.path.exists(dir_path):
            print(f"✗ Missing directory: {dir_path}")
            return False

    # Check files
    for file in required_files:
        file_path = os.path.join(base_path, file)
        if not os.path.exists(file_path):
            print(f"✗ Missing file: {file_path}")
            return False

    # Check that there are images in the Training directory
    training_dir = os.path.join(base_path, "Training")
    image_count = len([f for f in os.listdir(training_dir)
                      if f.endswith('.png') or f.endswith('.jpg')])

    if image_count == 0:
        print(f"✗ No images found in {training_dir}")
        return False

    print(f"✓ Data structure verified ({image_count} training images found)")
    return True

def main():
    parser = argparse.ArgumentParser(description='Setup and train OCR model')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples to generate (default: 100)')
    parser.add_argument('--no-clear', action='store_true',
                        help='Do not clear existing data')
    parser.add_argument('--skip-data', action='store_true',
                        help='Skip data generation and use existing data')

    args = parser.parse_args()

    # Start timer
    start_time = time.time()

    # Print header
    print_section_header("OCR Model Setup and Training")
    print("This script will set up sample data and train the OCR model")

    # Check dependencies
    if not check_python_dependencies():
        sys.exit(1)

    # Create sample data
    if not args.skip_data:
        if not create_sample_data(args.samples, not args.no_clear):
            sys.exit(1)
    else:
        print("\nSkipping data generation, using existing data...")
        # Still verify data structure
        if not verify_data_structure():
            print("Existing data structure is invalid.")
            sys.exit(1)

    # Train model
    if not train_model():
        sys.exit(1)

    # Done
    elapsed_time = time.time() - start_time
    print_section_header("Setup and Training Completed")
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    print(f"Model files saved in the 'models/' directory")

if __name__ == "__main__":
    main()
