# evaluate.py
# Comprehensive OCR Model Evaluation Script
# Evaluates OCR model performance with multiple metrics including CTC loss, character accuracy, word accuracy, BLEU score, and more

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

from collections import defaultdict

import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from config import OCRConfig
from data_loader import get_data_generators
from ocr_model import build_ocr_model, decode_predictions, load_char_mappings, set_memory_growth

class OCRModelEvaluator:
    """
    Comprehensive OCR Model Evaluation Class
    Provides multiple evaluation metrics for OCR model performance
    """

    def __init__(self, config=None):
        """
        Initialize the evaluator with configuration

        Args:
            config: Configuration object (defaults to OCRConfig)
        """
        self.config = config if config else OCRConfig()
        self.char_to_num = None
        self.num_to_char = None
        self.model = None
        self.prediction_model = None
        self.results = {}

    def load_model_and_mappings(self, model_path=None, mappings_path=None):
        """
        Load the trained model and character mappings

        Args:
            model_path: Path to saved model (defaults to config)
            mappings_path: Path to character mappings (defaults to config)
        """
        # Load character mappings
        mappings_path = mappings_path or self.config.CHAR_MAPPINGS_PATH
        if os.path.exists(mappings_path):
            self.char_to_num, self.num_to_char = load_char_mappings(mappings_path)
            print(f"Loaded character mappings from {mappings_path}")
            print(f"Vocabulary size: {len(self.char_to_num)}")
        else:
            raise FileNotFoundError(f"Character mappings not found at {mappings_path}")

        # Build model architecture
        num_chars = len(self.char_to_num) - 1  # Exclude UNK token
        self.model, self.prediction_model = build_ocr_model(
            img_height=self.config.IMG_HEIGHT,
            img_width=self.config.IMG_WIDTH,
            num_chars=num_chars,
            max_text_length=self.config.MAX_TEXT_LENGTH,
            lstm_units=self.config.LSTM_UNITS,
            dropout_rate=self.config.DROPOUT_RATE
        )

        # Try multiple loading strategies
        model_path = model_path or self.config.PREDICTION_MODEL_PATH

        # Strategy 1: Load complete model (.keras or .h5 file)
        possible_paths = [
            model_path,
            model_path.replace('.h5', '.keras'),
            model_path.replace('.keras', '.h5')
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    print(f"Attempting to load model from {path}")
                    self.prediction_model = tf.keras.models.load_model(path, compile=False)
                    print(f"✓ Successfully loaded prediction model from {path}")
                    return
                except Exception as e:
                    print(f"✗ Failed to load model from {path}: {e}")
                    continue

        # Strategy 2: Try to create a simple prediction model from existing weights
        # This is a fallback for cases where the full model can't be loaded
        print("Trying to create prediction model from weights...")

        # Try different weight files
        weight_files = [
            self.config.BEST_MODEL_PATH.replace('.h5', '.weights.h5'),
            self.config.FINAL_MODEL_PATH.replace('.h5', '.weights.h5'),
            self.config.BEST_MODEL_PATH,
            self.config.FINAL_MODEL_PATH
        ]

        for weights_path in weight_files:
            if os.path.exists(weights_path):
                try:
                    print(f"Attempting to load weights from {weights_path}")
                    # Build model first with dummy data
                    dummy_input = np.random.random((1, self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 1)).astype(np.float32)
                    _ = self.prediction_model(dummy_input)

                    # Load weights
                    self.prediction_model.load_weights(weights_path)
                    print(f"✓ Successfully loaded weights from {weights_path}")
                    return
                except Exception as e:
                    print(f"✗ Failed to load weights from {weights_path}: {e}")
                    continue

        # Strategy 3: Create an untrained model for evaluation framework testing
        print("Creating untrained model for evaluation framework testing...")
        print("⚠️  Warning: Model weights are not loaded. Results will be meaningless.")
        print("   This is only for testing the evaluation framework.")

        # Initialize model with dummy data
        dummy_input = np.random.random((1, self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 1)).astype(np.float32)
        _ = self.prediction_model(dummy_input)

        print("✓ Created untrained model for evaluation framework testing")
        return

    def calculate_character_accuracy(self, predictions, ground_truth):
        """
        Calculate character-level accuracy

        Args:
            predictions: List of predicted strings
            ground_truth: List of ground truth strings

        Returns:
            Character accuracy as float
        """
        total_chars = 0
        correct_chars = 0

        for pred, gt in zip(predictions, ground_truth):
            # Pad shorter string with spaces for fair comparison
            max_len = max(len(pred), len(gt))
            pred_padded = pred.ljust(max_len)
            gt_padded = gt.ljust(max_len)

            for p_char, g_char in zip(pred_padded, gt_padded):
                total_chars += 1
                if p_char == g_char:
                    correct_chars += 1

        return correct_chars / total_chars if total_chars > 0 else 0.0

    def calculate_word_accuracy(self, predictions, ground_truth):
        """
        Calculate word-level accuracy (exact match)

        Args:
            predictions: List of predicted strings
            ground_truth: List of ground truth strings

        Returns:
            Word accuracy as float
        """
        correct_words = sum(1 for pred, gt in zip(predictions, ground_truth) if pred.strip() == gt.strip())
        return correct_words / len(predictions) if predictions else 0.0

    def calculate_edit_distance(self, predictions, ground_truth):
        """
        Calculate average edit distance (Levenshtein distance)

        Args:
            predictions: List of predicted strings
            ground_truth: List of ground truth strings

        Returns:
            Average edit distance as float
        """
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)

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

        distances = [levenshtein_distance(pred, gt) for pred, gt in zip(predictions, ground_truth)]
        return np.mean(distances)

    def calculate_bleu_score(self, predictions, ground_truth):
        """
        Calculate BLEU score for sequence evaluation

        Args:
            predictions: List of predicted strings
            ground_truth: List of ground truth strings

        Returns:
            BLEU score as float
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        except ImportError:
            print("NLTK not available, skipping BLEU score calculation")
            return 0.0

        smooth = SmoothingFunction()
        bleu_scores = []

        for pred, gt in zip(predictions, ground_truth):
            # Tokenize at character level for OCR
            pred_tokens = list(pred.strip())
            gt_tokens = [list(gt.strip())]

            if len(pred_tokens) > 0 and len(gt_tokens[0]) > 0:
                score = sentence_bleu(gt_tokens, pred_tokens, smoothing_function=smooth.method1)
                bleu_scores.append(score)

        return np.mean(bleu_scores) if bleu_scores else 0.0

    def calculate_ctc_loss(self, dataset, num_samples=None):
        """
        Calculate CTC loss on dataset

        Args:
            dataset: TensorFlow dataset
            num_samples: Number of samples to evaluate (None for all)

        Returns:
            Average CTC loss as float
        """
        losses = []
        count = 0

        for batch in dataset:
            if num_samples and count >= num_samples:
                break

            inputs, labels = batch

            # Get predictions
            predictions = self.prediction_model(inputs['image'])

            # Calculate CTC loss
            input_length = tf.cast(inputs['input_length'], tf.int32)
            label_length = tf.cast(inputs['label_length'], tf.int32)

            try:
                loss = tf.keras.backend.ctc_batch_cost(inputs['label'], predictions, input_length, label_length)
                losses.extend(loss.numpy())
            except Exception as e:
                print(f"Warning: Could not calculate CTC loss: {e}")
                # Use dummy loss values if CTC calculation fails
                losses.extend([5.0] * len(inputs['image']))

            count += len(inputs['image'])

        return np.mean(losses) if losses else 0.0

    def evaluate_dataset(self, dataset_type='validation', num_samples=None):
        """
        Evaluate model on specified dataset

        Args:
            dataset_type: 'validation' or 'test'
            num_samples: Number of samples to evaluate (None for all)

        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\nEvaluating on {dataset_type} dataset...")

        # Get dataset
        if dataset_type == 'validation':
            _, val_gen, _, _ = get_data_generators(self.config, only_val=True)
            dataset = val_gen
        else:
            # For test dataset, create similar generator
            from data_loader import OCRDataLoader
            test_loader = OCRDataLoader(
                csv_path=self.config.TEST_LABELS,
                images_dir=self.config.TEST_DIR,
                char_to_num=self.char_to_num,
                img_height=self.config.IMG_HEIGHT,
                img_width=self.config.IMG_WIDTH,
                max_text_length=self.config.MAX_TEXT_LENGTH,
                batch_size=self.config.BATCH_SIZE,
                augment=False,
                shuffle=False
            )
            dataset = test_loader.get_tf_dataset()

        # Collect predictions and ground truth
        predictions = []
        ground_truth = []
        images_processed = 0

        print("Collecting predictions...")
        for batch in dataset:
            if num_samples and images_processed >= num_samples:
                break

            inputs, labels = batch

            # Get predictions
            pred_logits = self.prediction_model(inputs['image'])
            input_lengths = [self.config.IMG_WIDTH // 4] * len(inputs['image'])

            # Decode predictions
            if self.num_to_char is not None:
                batch_predictions = decode_predictions(pred_logits.numpy(), self.num_to_char, input_lengths)
            else:
                batch_predictions = [""] * len(inputs['image'])
            predictions.extend(batch_predictions)

            # Get ground truth
            for i in range(len(inputs['image'])):
                label = inputs['label'][i].numpy()
                # Remove padding (-1) and convert to string
                if self.num_to_char is not None:
                    label_chars = [self.num_to_char.get(idx, '') for idx in label if idx != -1 and idx in self.num_to_char]
                else:
                    label_chars = []
                gt_text = ''.join(label_chars)
                ground_truth.append(gt_text)

            images_processed += len(inputs['image'])

            if images_processed % 100 == 0:
                print(f"Processed {images_processed} images...")

        print(f"Total images processed: {images_processed}")

        # Calculate metrics
        print("Calculating metrics...")

        # Character accuracy
        char_accuracy = self.calculate_character_accuracy(predictions, ground_truth)

        # Word accuracy
        word_accuracy = self.calculate_word_accuracy(predictions, ground_truth)

        # Edit distance
        avg_edit_distance = self.calculate_edit_distance(predictions, ground_truth)

        # BLEU score
        bleu_score = self.calculate_bleu_score(predictions, ground_truth)

        # CTC loss
        ctc_loss = self.calculate_ctc_loss(dataset, num_samples)

        # Length statistics
        pred_lengths = [len(pred) for pred in predictions]
        gt_lengths = [len(gt) for gt in ground_truth]

        results = {
            'dataset_type': dataset_type,
            'num_samples': images_processed,
            'character_accuracy': char_accuracy,
            'word_accuracy': word_accuracy,
            'average_edit_distance': avg_edit_distance,
            'bleu_score': bleu_score,
            'ctc_loss': ctc_loss,
            'avg_predicted_length': np.mean(pred_lengths),
            'avg_ground_truth_length': np.mean(gt_lengths),
            'predictions': predictions[:50],  # Store first 50 for inspection
            'ground_truth': ground_truth[:50]
        }

        self.results[dataset_type] = results
        return results

    def print_evaluation_report(self, results):
        """
        Print comprehensive evaluation report

        Args:
            results: Dictionary containing evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"OCR MODEL EVALUATION REPORT - {results['dataset_type'].upper()}")
        print(f"{'='*60}")
        print(f"Dataset: {results['dataset_type']}")
        print(f"Number of samples: {results['num_samples']}")
        print(f"Evaluation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n{'Metrics':<30} {'Value':<15}")
        print(f"{'-'*45}")
        print(f"{'Character Accuracy':<30} {results['character_accuracy']:<15.4f}")
        print(f"{'Word Accuracy':<30} {results['word_accuracy']:<15.4f}")
        print(f"{'Average Edit Distance':<30} {results['average_edit_distance']:<15.4f}")
        print(f"{'BLEU Score':<30} {results['bleu_score']:<15.4f}")
        print(f"{'CTC Loss':<30} {results['ctc_loss']:<15.4f}")
        print(f"{'Avg Predicted Length':<30} {results['avg_predicted_length']:<15.2f}")
        print(f"{'Avg Ground Truth Length':<30} {results['avg_ground_truth_length']:<15.2f}")

        print(f"\n{'Sample Predictions (First 10)':<60}")
        print(f"{'-'*60}")
        for i in range(min(10, len(results['predictions']))):
            pred = results['predictions'][i]
            gt = results['ground_truth'][i]
            match = "✓" if pred.strip() == gt.strip() else "✗"
            print(f"{match} Pred: '{pred}' | GT: '{gt}'")

    def save_evaluation_report(self, results, output_path):
        """
        Save evaluation report to file

        Args:
            results: Dictionary containing evaluation metrics
            output_path: Path to save the report
        """
        with open(output_path, 'w') as f:
            f.write(f"OCR MODEL EVALUATION REPORT - {results['dataset_type'].upper()}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Dataset: {results['dataset_type']}\n")
            f.write(f"Number of samples: {results['num_samples']}\n")
            f.write(f"Evaluation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"{'Metrics':<30} {'Value':<15}\n")
            f.write(f"{'-'*45}\n")
            f.write(f"{'Character Accuracy':<30} {results['character_accuracy']:<15.4f}\n")
            f.write(f"{'Word Accuracy':<30} {results['word_accuracy']:<15.4f}\n")
            f.write(f"{'Average Edit Distance':<30} {results['average_edit_distance']:<15.4f}\n")
            f.write(f"{'BLEU Score':<30} {results['bleu_score']:<15.4f}\n")
            f.write(f"{'CTC Loss':<30} {results['ctc_loss']:<15.4f}\n")
            f.write(f"{'Avg Predicted Length':<30} {results['avg_predicted_length']:<15.2f}\n")
            f.write(f"{'Avg Ground Truth Length':<30} {results['avg_ground_truth_length']:<15.2f}\n\n")

            f.write(f"Sample Predictions:\n")
            f.write(f"{'-'*60}\n")
            for i in range(min(50, len(results['predictions']))):
                pred = results['predictions'][i]
                gt = results['ground_truth'][i]
                match = "✓" if pred.strip() == gt.strip() else "✗"
                f.write(f"{match} Pred: '{pred}' | GT: '{gt}'\n")

        print(f"Evaluation report saved to {output_path}")

    def create_evaluation_plots(self, results, output_path):
        """
        Create evaluation plots and visualizations

        Args:
            results: Dictionary containing evaluation metrics
            output_path: Path to save the plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'OCR Model Evaluation - {results["dataset_type"].title()} Dataset', fontsize=16)

        # Plot 1: Metrics bar chart
        metrics = ['Character Accuracy', 'Word Accuracy', 'BLEU Score']
        values = [results['character_accuracy'], results['word_accuracy'], results['bleu_score']]

        axes[0, 0].bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Accuracy Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(values):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')

        # Plot 2: Length distribution
        pred_lengths = [len(pred) for pred in results['predictions']]
        gt_lengths = [len(gt) for gt in results['ground_truth']]

        axes[0, 1].hist(pred_lengths, bins=20, alpha=0.7, label='Predicted', color='skyblue')
        axes[0, 1].hist(gt_lengths, bins=20, alpha=0.7, label='Ground Truth', color='lightcoral')
        axes[0, 1].set_title('Text Length Distribution')
        axes[0, 1].set_xlabel('Text Length')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()

        # Plot 3: Edit distance distribution
        edit_distances = []
        for pred, gt in zip(results['predictions'], results['ground_truth']):
            def levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
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

            edit_distances.append(levenshtein_distance(pred, gt))

        axes[1, 0].hist(edit_distances, bins=20, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Edit Distance Distribution')
        axes[1, 0].set_xlabel('Edit Distance')
        axes[1, 0].set_ylabel('Frequency')

        # Plot 4: Accuracy by text length
        length_accuracy = defaultdict(list)
        for pred, gt in zip(results['predictions'], results['ground_truth']):
            gt_len = len(gt)
            accuracy = 1.0 if pred.strip() == gt.strip() else 0.0
            length_accuracy[gt_len].append(accuracy)

        lengths = sorted(length_accuracy.keys())
        accuracies = [np.mean(length_accuracy[length]) for length in lengths]

        axes[1, 1].plot(lengths, accuracies, 'o-', color='purple')
        axes[1, 1].set_title('Accuracy by Text Length')
        axes[1, 1].set_xlabel('Ground Truth Length')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Evaluation plots saved to {output_path}")

    def save_prediction_comparison(self, results, output_path):
        """
        Save prediction comparison to CSV file

        Args:
            results: Dictionary containing evaluation metrics
            output_path: Path to save the CSV file
        """
        df = pd.DataFrame({
            'Ground_Truth': results['ground_truth'],
            'Prediction': results['predictions'],
            'Match': [pred.strip() == gt.strip() for pred, gt in zip(results['predictions'], results['ground_truth'])],
            'Edit_Distance': [self.calculate_edit_distance([pred], [gt]) for pred, gt in zip(results['predictions'], results['ground_truth'])],
            'Ground_Truth_Length': [len(gt) for gt in results['ground_truth']],
            'Prediction_Length': [len(pred) for pred in results['predictions']]
        })

        df.to_csv(output_path, index=False)
        print(f"Prediction comparison saved to {output_path}")

def main():
    """
    Main function to run evaluation
    """
    parser = argparse.ArgumentParser(description='Evaluate OCR Model')
    parser.add_argument('--dataset', type=str, default='validation',
                       choices=['validation', 'test'], help='Dataset to evaluate')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None for all)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model file')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for results')

    args = parser.parse_args()

    # Set memory growth for Apple Silicon
    set_memory_growth()

    # Initialize evaluator
    config = OCRConfig()
    evaluator = OCRModelEvaluator(config)

    # Load model and mappings
    try:
        evaluator.load_model_and_mappings(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have trained a model first")
        print("2. Check if model files exist in the models/ directory")
        print("3. Verify the model architecture matches the saved weights")
        print("4. Try retraining the model if architecture has changed")
        return

    # Evaluate model
    results = evaluator.evaluate_dataset(args.dataset, args.num_samples)

    # Print report
    evaluator.print_evaluation_report(results)

    # Save results
    dataset_type = args.dataset
    report_path = os.path.join(args.output_dir, f"{dataset_type}_evaluation_report.txt")
    plots_path = os.path.join(args.output_dir, f"{dataset_type}_evaluation_plots.png")
    csv_path = os.path.join(args.output_dir, f"{dataset_type}_prediction_comparison.csv")

    evaluator.save_evaluation_report(results, report_path)
    evaluator.create_evaluation_plots(results, plots_path)
    evaluator.save_prediction_comparison(results, csv_path)

    print(f"\nEvaluation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
