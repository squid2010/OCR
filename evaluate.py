# OCR/evaluate.py
"""
OCR Model Evaluation Module

Provides:
- OCRModelEvaluator: Class for evaluating OCR models (character/word accuracy, edit distance, BLEU)
- Exposes evaluate_model() for calling from other scripts
- CLI usage: python3 evaluate.py --num_samples 100

Refactored to use predict.py for making predictions (with snapping logic).
"""

import os
import numpy as np
import tensorflow as tf
import argparse
import pandas as pd
import matplotlib.pyplot as plt

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

from ocr_model import build_ocr_model, decode_predictions, load_char_mappings
from data_loader import get_data_generators
from predict import predict_medicine_name

class OCRModelEvaluator:
    def __init__(self, config=None):
        self.config = config
        self.char_to_num = None
        self.num_to_char = None
        self.prediction_model = None

    def load_model_and_mappings(self, model_path=None, mappings_path=None):
        # For compatibility, still load char mappings, but prediction will use predict.py
        mappings_path = mappings_path or "models/char_mappings.pkl"
        if not os.path.exists(mappings_path):
            raise FileNotFoundError(f"Character mappings not found at {mappings_path}")
        self.char_to_num, self.num_to_char = load_char_mappings(mappings_path)
        print(f"Loaded character mappings from {mappings_path} (vocab size: {len(self.char_to_num)})")

        # No longer loading valid medicine names for snapping

    def calculate_character_accuracy(self, predictions, ground_truth):
        total_chars = 0
        correct_chars = 0
        for pred, gt in zip(predictions, ground_truth):
            max_len = max(len(pred), len(gt))
            pred_padded = pred.ljust(max_len)
            gt_padded = gt.ljust(max_len)
            for p_char, g_char in zip(pred_padded, gt_padded):
                total_chars += 1
                if p_char == g_char:
                    correct_chars += 1
        return correct_chars / total_chars if total_chars > 0 else 0.0

    def calculate_word_accuracy(self, predictions, ground_truth):
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p.strip() == g.strip())
        return correct / len(predictions) if predictions else 0.0

    def calculate_edit_distances(self, predictions, ground_truth):
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
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
        distances = [levenshtein(p, t) for p, t in zip(predictions, ground_truth)]
        return distances

    def calculate_bleu(self, predictions, ground_truth):
        if not BLEU_AVAILABLE:
            print("nltk BLEU not available. Skipping BLEU computation.")
            return None
        smooth = SmoothingFunction().method1
        scores = []
        for pred, ref in zip(predictions, ground_truth):
            ref_tokens = list(ref.strip())
            pred_tokens = list(pred.strip())
            score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
            scores.append(score)
        return np.mean(scores)

    def run(self, num_samples=None):
        import numpy as np
        print("Loading testing dataset...")
        import pandas as pd
        from PIL import Image
        test_csv = "trainingData/Testing/testing_labels.csv"
        test_img_dir = "trainingData/Testing/testing_words"
        df = pd.read_csv(test_csv)
        image_paths = df["IMAGE"].astype(str).tolist()
        # Use MEDICINE_NAME unless it's BOTTOM_BOX, then use GENERIC_NAME
        def get_label_for_row(row):
            if "MEDICINE_NAME" in row and str(row["MEDICINE_NAME"]) == "BOTTOM_BOX":
                return str(row.get("GENERIC_NAME", ""))
            else:
                return str(row.get("MEDICINE_NAME", ""))
        labels = [get_label_for_row(row) for _, row in df.iterrows()]
        # Build vocab from all labels
        chars = set()
        for label in labels:
            chars.update(label)
        vocab = ['UNK'] + sorted(list(chars))
        char_to_num = {c: i for i, c in enumerate(vocab)}
        num_to_char = {i: c for i, c in enumerate(vocab)}
        self.char_to_num = char_to_num
        self.num_to_char = num_to_char
        # Preprocess all images
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(os.path.join(test_img_dir, img_path)).convert("L")
                img = np.array(img)
            except Exception:
                img = np.ones((128, 384), dtype=np.uint8) * 255
            # Resize and pad
            h, w = img.shape
            img_height, img_width = 128, 384
            scale = min(img_width / w, img_height / h)
            nw, nh = int(w * scale), int(h * scale)
            from PIL import Image as PILImage
            img_resized = PILImage.fromarray(img).resize((nw, nh), resample=PILImage.BILINEAR)
            img_resized = np.array(img_resized)
            canvas = np.ones((img_height, img_width), dtype=np.uint8) * 255
            x_offset = (img_width - nw) // 2
            y_offset = (img_height - nh) // 2
            canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = img_resized
            img = canvas.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
            images.append(img)
        images = np.stack(images)
        # Encode all labels
        def encode_label(label, max_text_length=32):
            label = label[:max_text_length]
            label_encoded = [char_to_num.get(c, char_to_num['UNK']) for c in label]
            label_encoded += [-1] * (max_text_length - len(label_encoded))
            return np.array(label_encoded, dtype=np.int32)
        label_indices = np.stack([encode_label(label) for label in labels])
        ground_truth_texts = labels

        # Use predict.py for predictions (with snapping, numpy array API) - batch all at once
        import numpy as np
        imgs_np = images.numpy() if hasattr(images, 'numpy') else images

        # Load model and mappings once
        from predict import load_char_mappings
        from tensorflow.keras.models import load_model
        import tensorflow.keras.backend as K

        model = load_model("models/ocr_model_final.keras", compile=False)
        char_to_num, num_to_char = load_char_mappings("models/char_mappings.pkl")

        # Predict all at once
        preds = model.predict(imgs_np)
        decoded, _ = K.ctc_decode(preds, input_length=np.ones(preds.shape[0]) * preds.shape[1], greedy=True)
        decoded_indices = K.get_value(decoded[0])

        decoded_preds = []
        for i, seq in enumerate(decoded_indices):
            medicine_name = ''.join([num_to_char.get(idx, '') for idx in seq if idx != -1])
            decoded_preds.append(medicine_name)

        char_acc = self.calculate_character_accuracy(decoded_preds, ground_truth_texts)
        word_acc = self.calculate_word_accuracy(decoded_preds, ground_truth_texts)
        edit_distances = self.calculate_edit_distances(decoded_preds, ground_truth_texts)
        avg_edit_distance = np.mean(edit_distances)
        bleu_score = self.calculate_bleu(decoded_preds, ground_truth_texts)

        print("\nEvaluation Results:")
        print(f"Character Accuracy: {char_acc:.4f}")
        print(f"Word Accuracy:      {word_acc:.4f}")
        print(f"Avg Edit Distance:  {avg_edit_distance:.4f}")
        if bleu_score is not None:
            print(f"BLEU Score:         {bleu_score:.4f}")

        # Save outputs
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)

        # CSV with predictions
        df = pd.DataFrame({
            "GroundTruth": ground_truth_texts,
            "Prediction": decoded_preds,
            "EditDistance": edit_distances
        })
        csv_path = os.path.join(results_dir, "predictions.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved prediction CSV to {csv_path}")

        # Text summary
        summary_path = os.path.join(results_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Character Accuracy: {char_acc:.4f}\n")
            f.write(f"Word Accuracy:      {word_acc:.4f}\n")
            f.write(f"Avg Edit Distance:  {avg_edit_distance:.4f}\n")
            if bleu_score is not None:
                f.write(f"BLEU Score:         {bleu_score:.4f}\n")
        print(f"Saved summary to {summary_path}")

        # Edit distance histogram
        plt.figure(figsize=(6,4))
        plt.hist(edit_distances, bins=range(0, max(edit_distances)+2), color="skyblue", edgecolor="black")
        plt.xlabel("Edit Distance")
        plt.ylabel("Count")
        plt.title("Edit Distance Distribution")
        hist_path = os.path.join(results_dir, "edit_distance_histogram.png")
        plt.savefig(hist_path)
        plt.close()
        print(f"Saved histogram to {hist_path}")

        return {
            "character_accuracy": char_acc,
            "word_accuracy": word_acc,
            "edit_distance": avg_edit_distance,
            "bleu_score": bleu_score
        }

def evaluate_model(num_samples=None):
    evaluator = OCRModelEvaluator()
    evaluator.load_model_and_mappings()
    return evaluator.run(num_samples=num_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OCR model performance.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    args = parser.parse_args()
    evaluate_model(num_samples=args.num_samples)
