import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import Sequence
from PIL import Image, ImageOps
import random

# =========================
# Data Loader & Preprocessor
# =========================

class OCRDataLoader:
    def __init__(
        self,
        csv_path,
        images_dir,
        char_to_num,
        label_column="MEDICINE_NAME",  # NEW: which column to use for labels
        img_height=128,
        img_width=384,
        max_text_length=32,
        batch_size=16,
        augment=False,
        shuffle=True,
        seed=42,
        skip_empty=True,  # NEW: skip empty boxes
        skip_bottom_box=True,  # NEW: skip bottom box rows for handwritten OCR
    ):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.char_to_num = char_to_num
        self.label_column = label_column
        self.img_height = img_height
        self.img_width = img_width
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.seed = seed
        self.skip_empty = skip_empty
        self.skip_bottom_box = skip_bottom_box

        self.df = pd.read_csv(csv_path)
        # Filter out samples with labels longer than max_text_length
        filtered = self.df[self.df[self.label_column].astype(str).str.len() <= self.max_text_length]
        # Optionally skip empty boxes
        if self.skip_empty and "EMPTY" in filtered.columns:
            filtered = filtered[filtered["EMPTY"] == False]
        # Optionally skip bottom box rows (if label_column is MEDICINE_NAME)
        if self.skip_bottom_box and self.label_column == "MEDICINE_NAME" and "MEDICINE_NAME" in filtered.columns:
            filtered = filtered[filtered["MEDICINE_NAME"] != "BOTTOM_BOX"]
        self.image_paths = filtered['IMAGE'].astype(str).tolist()
        self.labels = filtered[self.label_column].astype(str).tolist()
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def _augment_image(self, img):
        # Random brightness
        if random.random() < 0.3:
            factor = 0.7 + 0.6 * random.random()
            img = Image.fromarray(img)
            img = ImageOps.autocontrast(img)
            img = np.array(img) * factor
            img = np.clip(img, 0, 255).astype(np.uint8)
        # Random rotation
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)
            img = Image.fromarray(img)
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
            img = np.array(img)
        # Random noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 8, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Random erasing
        if random.random() < 0.1:
            x = random.randint(0, self.img_width - 10)
            y = random.randint(0, self.img_height - 10)
            w = random.randint(5, 20)
            h = random.randint(5, 20)
            img[y:y+h, x:x+w] = 255
        return img

    def _preprocess_image(self, img_path):
        # Read image as grayscale using PIL
        try:
            img = Image.open(os.path.join(self.images_dir, img_path)).convert("L")
            img = np.array(img)
        except Exception:
            img = np.ones((self.img_height, self.img_width), dtype=np.uint8) * 255
        # Resize and pad
        h, w = img.shape
        scale = min(self.img_width / w, self.img_height / h)
        nw, nh = int(w * scale), int(h * scale)
        img_resized = Image.fromarray(img).resize((nw, nh), resample=Image.BILINEAR)
        img_resized = np.array(img_resized)
        canvas = np.ones((self.img_height, self.img_width), dtype=np.uint8) * 255
        x_offset = (self.img_width - nw) // 2
        y_offset = (self.img_height - nh) // 2
        canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = img_resized
        img = canvas
        if self.augment:
            img = self._augment_image(img)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        return img

    def _encode_label(self, label):
        # Convert label string to list of int indices
        label = label[:self.max_text_length]
        label_encoded = [self.char_to_num.get(c, self.char_to_num['UNK']) for c in label]
        # Pad with -1 (CTC blank)
        label_encoded += [-1] * (self.max_text_length - len(label_encoded))
        return np.array(label_encoded, dtype=np.int32)

    def generator(self):
        while True:
            if self.shuffle:
                np.random.shuffle(self.indices)
            for start in range(0, len(self.indices), self.batch_size):
                batch_indices = self.indices[start:start+self.batch_size]
                batch_images = []
                batch_labels = []
                input_lengths = []
                label_lengths = []
                for idx in batch_indices:
                    img = self._preprocess_image(self.image_paths[idx])
                    label = self._encode_label(self.labels[idx])
                    batch_images.append(img)
                    batch_labels.append(label)
                    input_lengths.append(self.img_width // 4)  # Downsampled by CNN
                    label_lengths.append(min(len(self.labels[idx]), self.max_text_length))
                # Skip empty batches to avoid CTC errors
                if len(batch_images) == 0:
                    continue
                batch_images = np.array(batch_images)
                batch_labels = np.array(batch_labels)
                input_lengths = np.array(input_lengths)
                label_lengths = np.array(label_lengths)
                # For standard Keras fit: yield (images, labels)
                yield (np.array(batch_images), np.array(batch_labels))

    def get_tf_dataset(self):
        output_signature = (
            tf.TensorSpec(shape=(None, self.img_height, self.img_width, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, self.max_text_length), dtype=tf.int32),
        )
        ds = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=output_signature,
        )
        ds = ds.prefetch(buffer_size=1)
        return ds

# =========================
# Utility: Build Vocabulary
# =========================

def build_vocab_from_labels(csv_paths, label_column="MEDICINE_NAME", extra_tokens=['UNK']):
    chars = set()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        # Optionally skip bottom box rows
        if label_column == "MEDICINE_NAME" and "MEDICINE_NAME" in df.columns:
            df = df[df["MEDICINE_NAME"] != "BOTTOM_BOX"]
        for label in df[label_column].astype(str):
            chars.update(label)
    chars = sorted(list(chars))
    vocab = extra_tokens + chars
    char_to_num = {c: i for i, c in enumerate(vocab)}
    num_to_char = {i: c for i, c in enumerate(vocab)}
    return char_to_num, num_to_char, vocab

# =========================
# Data Generator Utilities for Training/Eval
# =========================

def get_data_generators(config, only_val=False, label_column="MEDICINE_NAME", skip_empty=True, skip_bottom_box=True):
    """
    Returns:
        train_gen, val_gen, char_to_num, num_to_char
    If only_val=True, train_gen is None.
    """
    train_csv = getattr(config, "TRAIN_LABELS", "trainingData/Training/training_labels.csv")
    val_csv = getattr(config, "VAL_LABELS", "trainingData/Validation/validation_labels.csv")
    train_img_dir = getattr(config, "TRAIN_DIR", "trainingData/Training/training_words")
    val_img_dir = getattr(config, "VAL_DIR", "trainingData/Validation/validation_words")
    img_height = getattr(config, "IMG_HEIGHT", 128)
    img_width = getattr(config, "IMG_WIDTH", 384)
    max_text_length = getattr(config, "MAX_TEXT_LENGTH", 32)
    batch_size = getattr(config, "BATCH_SIZE", 8)
    use_aug = getattr(config, "USE_AUGMENTATION", True)

    # Build vocab from both train and val
    char_to_num, num_to_char, vocab = build_vocab_from_labels([train_csv, val_csv], label_column=label_column)

    val_loader = OCRDataLoader(
        csv_path=val_csv,
        images_dir=val_img_dir,
        char_to_num=char_to_num,
        label_column=label_column,
        img_height=img_height,
        img_width=img_width,
        max_text_length=max_text_length,
        batch_size=batch_size,
        augment=False,
        shuffle=False,
        skip_empty=skip_empty,
        skip_bottom_box=skip_bottom_box,
    )
    val_gen = val_loader.get_tf_dataset()

    if only_val:
        return None, val_gen, char_to_num, num_to_char

    train_loader = OCRDataLoader(
        csv_path=train_csv,
        images_dir=train_img_dir,
        char_to_num=char_to_num,
        label_column=label_column,
        img_height=img_height,
        img_width=img_width,
        max_text_length=max_text_length,
        batch_size=batch_size,
        augment=use_aug,
        shuffle=True,
        skip_empty=skip_empty,
        skip_bottom_box=skip_bottom_box,
    )
    train_gen = train_loader.get_tf_dataset()

    return train_gen, val_gen, char_to_num, num_to_char

# =========================
# Single/Batch Image Loader Utilities
# =========================

def load_single_image(image_path, config):
    """
    Loads and preprocesses a single image for prediction.
    """
    img_height = getattr(config, "IMG_HEIGHT", 128)
    img_width = getattr(config, "IMG_WIDTH", 384)
    try:
        img = Image.open(image_path).convert("L")
        img = np.array(img)
    except Exception:
        img = np.ones((img_height, img_width), dtype=np.uint8) * 255
    h, w = img.shape
    scale = min(img_width / w, img_height / h)
    nw, nh = int(w * scale), int(h * scale)
    img_resized = Image.fromarray(img).resize((nw, nh), resample=Image.BILINEAR)
    img_resized = np.array(img_resized)
    canvas = np.ones((img_height, img_width), dtype=np.uint8) * 255
    x_offset = (img_width - nw) // 2
    y_offset = (img_height - nh) // 2
    canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = img_resized
    img = canvas.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def load_batch_images(folder_path, config):
    """
    Loads and preprocesses all images in a folder for batch prediction.
    Returns a numpy array of shape (N, H, W, 1)
    """
    img_height = getattr(config, "IMG_HEIGHT", 128)
    img_width = getattr(config, "IMG_WIDTH", 384)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    images = []
    for fname in image_files:
        img_path = os.path.join(folder_path, fname)
        img = load_single_image(img_path, config)
        images.append(img)
    return np.stack(images, axis=0), image_files
