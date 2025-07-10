import os
import numpy as np
import tensorflow as tf
import pandas as pd

from config import OCRConfig

IMG_HEIGHT = OCRConfig.IMG_HEIGHT
IMG_WIDTH = OCRConfig.IMG_WIDTH

def get_char_maps(labels):
    unique_chars = sorted(set(''.join(labels)))
    char_to_num = {c: i for i, c in enumerate(unique_chars)}  # 0-based, no blank
    num_to_char = {i: c for c, i in char_to_num.items()}
    return char_to_num, num_to_char

def encode_single_label(label, char_to_num, max_length):
    label = label[:max_length]
    encoded = [char_to_num.get(c, 0) for c in label]
    while len(encoded) < max_length:
        encoded.append(0)
    return np.array(encoded, dtype=np.int32)

def encode_labels(labels, char_to_num, max_length):
    encoded = []
    for label in labels:
        encoded.append(encode_single_label(label, char_to_num, max_length))
    return np.array(encoded)

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [OCRConfig.IMG_HEIGHT, OCRConfig.IMG_WIDTH])
    img = img / 255.0
    return img

def lazy_load_csv_dataset(csv_path, img_folder, char_to_num, max_length, batch_size, target_column='MEDICINE_NAME', shuffle=True):
    """
    Lazily load CSV rows and images using tf.data.
    Only the CSV file is loaded into memory, not all images.
    """
    # Read CSV with pandas just to get the number of rows and char maps
    df = pd.read_csv(csv_path)
    num_samples = len(df)
    # Use tf.data.experimental.make_csv_dataset for lazy loading
    def _make_path(image, *args):
        # tf.py_function returns bytes, decode to str
        image = image.numpy().decode('utf-8')
        return os.path.join(img_folder, image)
    def _generator():
        for _, row in df.iterrows():
            img_path = os.path.join(img_folder, row['IMAGE'])
            label = str(row[target_column])
            yield img_path, label

    def _tf_encode(img_path, label):
        img = preprocess_image(img_path)
        label_encoded = tf.py_function(
            func=lambda x: encode_single_label(x.numpy().decode('utf-8'), char_to_num, max_length),
            inp=[label],
            Tout=tf.int32
        )
        label_encoded.set_shape([max_length])
        return img, label_encoded

    dataset = tf.data.Dataset.from_generator(
        _generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
    )
    dataset = dataset.map(_tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1000, num_samples))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

# The above functions are replaced by lazy_load_csv_dataset for training.

def load_csv_labels(csv_path, img_folder):
    """
    Loads image paths and labels from a CSV file.
    Returns: (image_paths, medicine_names, generic_names)
    """
    df = pd.read_csv(csv_path)
    image_paths = [os.path.join(img_folder, row['IMAGE']) for _, row in df.iterrows()]
    medicine_names = df['MEDICINE_NAME'].astype(str).tolist()
    generic_names = df['GENERIC_NAME'].astype(str).tolist()
    return image_paths, medicine_names, generic_names
