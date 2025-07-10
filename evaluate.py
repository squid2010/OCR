import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from ocr_model import ctc_lambda_func, build_ocr_model
from config import OCRConfig
from data_loader import load_data, preprocess_image, create_char_mappings
from predict import decode_batch_predictions
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from tqdm import tqdm

def levenshtein_distance(s1, s2):
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j
