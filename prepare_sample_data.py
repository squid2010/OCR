#!/usr/bin/env python3
"""
Utility script to create sample data for the OCR model training
This script generates sample images and CSV files with labels for training, validation, and testing
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import argparse
import string
import shutil

def generate_handwritten_like_text(text, width=384, height=128, background_color=(255, 255, 255), text_color=(0, 0, 0)):
    """Generate an image that simulates handwritten text"""
    img = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default if not available
    try:
        # Try to find a handwriting-like font
        font_paths = [
            '/Library/Fonts/Arial.ttf',  # macOS
            '/System/Library/Fonts/Supplemental/Arial.ttf',  # macOS
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
            'C:\\Windows\\Fonts\\arial.ttf'  # Windows
        ]

        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, size=48)
                break

        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Calculate text position to center it
    text_width = font.getbbox(text)[2] - font.getbbox(text)[0]
    text_height = font.getbbox(text)[3] - font.getbbox(text)[1]
    position = ((width - text_width) // 2, (height - text_height) // 2)

    # Apply a slight random rotation to simulate handwriting variation
    angle = random.uniform(-5, 5)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=0)
    draw = ImageDraw.Draw(img)

    # Draw the text
    draw.text(position, text, fill=text_color, font=font)

    # Convert to grayscale
    img = img.convert('L')

    # Add some noise to simulate real-world conditions
    img_array = np.array(img)
    noise = np.random.normal(0, 5, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img_array)

def create_sample_dataset(base_dir, num_samples=10, splits=(0.7, 0.2, 0.1)):
    """Create a sample dataset with the given number of samples"""

    # Create directories
    train_dir = os.path.join(base_dir, 'Training')
    val_dir = os.path.join(base_dir, 'Validation')
    test_dir = os.path.join(base_dir, 'Testing')

    # Create word directories
    train_words_dir = os.path.join(train_dir, 'training_words')
    val_words_dir = os.path.join(val_dir, 'validation_words')
    test_words_dir = os.path.join(test_dir, 'testing_words')

    for directory in [train_dir, val_dir, test_dir, train_words_dir, val_words_dir, test_words_dir]:
        os.makedirs(directory, exist_ok=True)

    # Sample data: pairs of medicine names and their generic names
    medicine_generic_pairs = [
        ('Aceta', 'Paracetamol'),
        ('Napa', 'Paracetamol'),
        ('Ace', 'Paracetamol'),
        ('Alatrol', 'Cetirizine Hydrochloride'),
        ('Amodis', 'Metronidazole'),
        ('Azithrocin', 'Azithromycin Dihydrate'),
        ('Azyth', 'Azithromycin Dihydrate'),
        ('Bacaid', 'Baclofen'),
        ('Baclofen', 'Baclofen'),
        ('Canazole', 'Fluconazole'),
        ('Cetisoft', 'Cetirizine Hydrochloride'),
        ('Denixil', 'Clonazepam'),
        ('Dinafex', 'Fexofenadine Hydrochloride'),
        ('Esonix', 'Esomeprazole'),
        ('Fexo', 'Fexofenadine Hydrochloride'),
        ('Maxpro', 'Esomeprazole'),
        ('Metro', 'Metronidazole'),
        ('Napa Extend', 'Paracetamol'),
        ('Rhinil', 'Cetirizine Hydrochloride'),
        ('Telfast', 'Fexofenadine Hydrochloride')
    ]

    # Calculate split counts
    train_count = int(num_samples * splits[0])
    val_count = int(num_samples * splits[1])
    test_count = num_samples - train_count - val_count

    print(f"Creating sample dataset with {num_samples} samples")
    print(f"Training: {train_count}, Validation: {val_count}, Testing: {test_count}")

    # Lists to store CSV data
    train_data = []
    val_data = []
    test_data = []

    # Generate samples
    for i in range(num_samples):
        # Choose a random medicine-generic pair
        medicine, generic = random.choice(medicine_generic_pairs)

        # Generate image filename
        img_filename = f"{i}.png"

        # Decide which split this sample goes into
        if i < train_count:
            target_dir = train_dir
            target_data = train_data
        elif i < train_count + val_count:
            target_dir = val_dir
            target_data = val_data
        else:
            target_dir = test_dir
            target_data = test_data

        # Generate the image
        img = generate_handwritten_like_text(medicine)

        # Save to the appropriate words directory
        if target_dir == train_dir:
            img_path = os.path.join(train_dir, 'training_words', img_filename)
        elif target_dir == val_dir:
            img_path = os.path.join(val_dir, 'validation_words', img_filename)
        else:
            img_path = os.path.join(test_dir, 'testing_words', img_filename)

        img.save(img_path)

        # Add to CSV data
        target_data.append({
            'IMAGE': img_filename,
            'MEDICINE_NAME': medicine,
            'GENERIC_NAME': generic
        })

    # Save CSV files
    pd.DataFrame(train_data).to_csv(os.path.join(train_dir, 'training_labels.csv'), index=False)
    pd.DataFrame(val_data).to_csv(os.path.join(val_dir, 'validation_labels.csv'), index=False)
    pd.DataFrame(test_data).to_csv(os.path.join(test_dir, 'testing_labels.csv'), index=False)

    print("Sample dataset created successfully!")
    print(f"Dataset directory: {base_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate sample data for OCR training')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of total samples to generate (default: 100)')
    parser.add_argument('--output', type=str, default='trainingData',
                        help='Output directory (default: trainingData)')
    parser.add_argument('--clear', action='store_true',
                        help='Clear existing data in the output directory')

    args = parser.parse_args()

    base_dir = args.output

    # Clear existing data if requested
    if args.clear and os.path.exists(base_dir):
        print(f"Clearing existing data in {base_dir}")
        shutil.rmtree(base_dir)

    # Create the sample dataset
    create_sample_dataset(base_dir, args.samples)

    print("\nTo train the model with this dataset, run:")
    print("python train.py")

if __name__ == "__main__":
    main()
