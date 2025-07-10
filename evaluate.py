import os
import tensorflow as tf
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import editdistance
from data_loader import load_csv_labels
from ocr_model import OCRModel, decode_predictions
from config import OCRConfig
import pandas as pd

class OCREvaluator:
    def __init__(self, model_path, char_mappings_path):
        """Initialize OCR evaluator with trained model and character mappings"""
        self.model_path = model_path
        self.char_mappings_path = char_mappings_path
        self.model = None
        self.char_to_num = None
        self.num_to_char = None
        self.load_model_and_mappings()

    def load_model_and_mappings(self):
        """Load trained model and character mappings"""
        print("Loading model and character mappings...")

        # Load character mappings
        with open(self.char_mappings_path, 'rb') as f:
            mappings = pickle.load(f)
            self.char_to_num = mappings['char_to_num']
            self.num_to_char = mappings['num_to_char']

        # Load model
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model and mappings loaded successfully!")

    def preprocess_image(self, image_path, img_height=None, img_width=None):
        """Preprocess image for prediction"""
        if img_height is None:
            img_height = OCRConfig.IMG_HEIGHT
        if img_width is None:
            img_width = OCRConfig.IMG_WIDTH
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [img_height, img_width])
        img = tf.cast(img, tf.float32) / 255.0
        return tf.expand_dims(img, 0)  # Add batch dimension

    def predict_text(self, image_path):
        """Predict text from image"""
        img = self.preprocess_image(image_path)
        predictions = self.model(img)
        decoded_text = self.decode_predictions(predictions)
        return decoded_text[0] if decoded_text else ""

    def decode_predictions(self, predictions):
        """Decode model predictions to text"""
        input_len = np.ones(predictions.shape[0]) * predictions.shape[1]

        # Use CTC greedy decoder
        results = tf.keras.backend.ctc_decode(
            predictions,
            input_length=input_len,
            greedy=True
        )[0][0]

        # Convert to text
        output_text = []
        for res in results.numpy():
            text = ''.join([self.num_to_char.get(idx, '') for idx in res if idx >= 0])
            # Remove consecutive duplicates
            if text:
                cleaned_text = text[0] if text else ""
                for i in range(1, len(text)):
                    if text[i] != text[i-1]:
                        cleaned_text += text[i]
                output_text.append(cleaned_text.strip())
            else:
                output_text.append('')

        return output_text

    def calculate_edit_distance(self, predicted, actual):
        """Calculate edit distance between predicted and actual text"""
        return editdistance.eval(predicted.lower(), actual.lower())

    def calculate_character_accuracy(self, predicted, actual):
        """Calculate character-level accuracy"""
        if len(actual) == 0:
            return 1.0 if len(predicted) == 0 else 0.0

        correct_chars = sum(1 for p, a in zip(predicted.lower(), actual.lower()) if p == a)
        return correct_chars / max(len(predicted), len(actual))

    def calculate_word_accuracy(self, predicted, actual):
        """Calculate word-level accuracy"""
        return 1.0 if predicted.lower().strip() == actual.lower().strip() else 0.0

    def evaluate_dataset(self, csv_path, img_folder, target_column='MEDICINE_NAME'):
        """Evaluate model on a dataset"""
        print(f"Evaluating on dataset: {csv_path}")

        # Load data
        images, medicine_names, generic_names = load_csv_labels(csv_path, img_folder)

        # Choose target labels
        if target_column == 'MEDICINE_NAME':
            labels = medicine_names
        else:
            labels = generic_names

        predictions = []
        edit_distances = []
        char_accuracies = []
        word_accuracies = []

        print(f"Processing {len(images)} images...")

        for i, (img_path, true_label) in enumerate(zip(images, labels)):
            if i % 100 == 0:
                print(f"Processed {i}/{len(images)} images")

            try:
                # Predict
                pred_text = self.predict_text(img_path)
                predictions.append(pred_text)

                # Calculate metrics
                edit_dist = self.calculate_edit_distance(pred_text, true_label)
                char_acc = self.calculate_character_accuracy(pred_text, true_label)
                word_acc = self.calculate_word_accuracy(pred_text, true_label)

                edit_distances.append(edit_dist)
                char_accuracies.append(char_acc)
                word_accuracies.append(word_acc)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                predictions.append("")
                edit_distances.append(len(true_label))
                char_accuracies.append(0.0)
                word_accuracies.append(0.0)

        # Calculate overall metrics
        avg_edit_distance = np.mean(edit_distances)
        avg_char_accuracy = np.mean(char_accuracies)
        avg_word_accuracy = np.mean(word_accuracies)

        results = {
            'predictions': predictions,
            'true_labels': labels,
            'edit_distances': edit_distances,
            'char_accuracies': char_accuracies,
            'word_accuracies': word_accuracies,
            'avg_edit_distance': avg_edit_distance,
            'avg_char_accuracy': avg_char_accuracy,
            'avg_word_accuracy': avg_word_accuracy,
            'total_samples': len(images)
        }

        return results

    def generate_evaluation_report(self, results, save_path='evaluation_report.txt'):
        """Generate detailed evaluation report"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("OCR Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total Samples: {results['total_samples']}\n")
            f.write(f"Average Edit Distance: {results['avg_edit_distance']:.4f}\n")
            f.write(f"Average Character Accuracy: {results['avg_char_accuracy']:.4f}\n")
            f.write(f"Average Word Accuracy: {results['avg_word_accuracy']:.4f}\n\n")

            # Distribution of accuracies
            word_acc_dist = Counter([round(acc, 1) for acc in results['word_accuracies']])
            f.write("Word Accuracy Distribution:\n")
            for acc, count in sorted(word_acc_dist.items()):
                f.write(f"  {acc:.1f}: {count} samples\n")
            f.write("\n")

            # Examples of predictions
            f.write("Sample Predictions:\n")
            f.write("-" * 30 + "\n")
            for i in range(min(20, len(results['predictions']))):
                f.write(f"True: {results['true_labels'][i]}\n")
                f.write(f"Pred: {results['predictions'][i]}\n")
                f.write(f"Edit Distance: {results['edit_distances'][i]}\n")
                f.write(f"Word Accuracy: {results['word_accuracies'][i]:.2f}\n")
                f.write("-" * 30 + "\n")

        print(f"Evaluation report saved to {save_path}")

    def plot_evaluation_metrics(self, results, save_path='evaluation_plots.png'):
        """Create visualization of evaluation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Word accuracy distribution
        axes[0, 0].hist(results['word_accuracies'], bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Word Accuracy Distribution')
        axes[0, 0].set_xlabel('Word Accuracy')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(results['avg_word_accuracy'], color='red', linestyle='--',
                          label=f'Mean: {results["avg_word_accuracy"]:.3f}')
        axes[0, 0].legend()

        # Character accuracy distribution
        axes[0, 1].hist(results['char_accuracies'], bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('Character Accuracy Distribution')
        axes[0, 1].set_xlabel('Character Accuracy')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(results['avg_char_accuracy'], color='red', linestyle='--',
                          label=f'Mean: {results["avg_char_accuracy"]:.3f}')
        axes[0, 1].legend()

        # Edit distance distribution
        axes[1, 0].hist(results['edit_distances'], bins=20, alpha=0.7, color='orange')
        axes[1, 0].set_title('Edit Distance Distribution')
        axes[1, 0].set_xlabel('Edit Distance')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(results['avg_edit_distance'], color='red', linestyle='--',
                          label=f'Mean: {results["avg_edit_distance"]:.3f}')
        axes[1, 0].legend()

        # Accuracy vs Edit Distance scatter plot
        axes[1, 1].scatter(results['edit_distances'], results['word_accuracies'], alpha=0.6)
        axes[1, 1].set_title('Word Accuracy vs Edit Distance')
        axes[1, 1].set_xlabel('Edit Distance')
        axes[1, 1].set_ylabel('Word Accuracy')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Evaluation plots saved to {save_path}")

    def compare_predictions(self, results, save_path='prediction_comparison.csv'):
        """Save detailed comparison of predictions vs actual labels"""
        df = pd.DataFrame({
            'True_Label': results['true_labels'],
            'Predicted_Label': results['predictions'],
            'Edit_Distance': results['edit_distances'],
            'Character_Accuracy': results['char_accuracies'],
            'Word_Accuracy': results['word_accuracies']
        })

        # Sort by word accuracy (worst first for easier debugging)
        df = df.sort_values('Word_Accuracy')
        df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"Detailed comparison saved to {save_path}")

        return df

    def analyze_common_errors(self, results, top_n=10):
        """Analyze most common prediction errors"""
        error_pairs = []
        for true_label, pred_label, word_acc in zip(results['true_labels'],
                                                   results['predictions'],
                                                   results['word_accuracies']):
            if word_acc < 1.0:  # Only incorrect predictions
                error_pairs.append((true_label, pred_label))

        error_counter = Counter(error_pairs)

        print(f"\nTop {top_n} Most Common Errors:")
        print("-" * 50)
        for (true_label, pred_label), count in error_counter.most_common(top_n):
            print(f"True: '{true_label}' -> Predicted: '{pred_label}' (Count: {count})")

        return error_counter

def main():
    """Main evaluation function"""

    # Paths
    model_path = OCRConfig.PREDICTION_MODEL_PATH
    char_mappings_path = OCRConfig.CHAR_MAPPINGS_PATH

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train the model first using train.py")
        return

    if not os.path.exists(char_mappings_path):
        print(f"Character mappings file not found: {char_mappings_path}")
        print("Please train the model first using train.py")
        return

    # Initialize evaluator
    evaluator = OCREvaluator(model_path, char_mappings_path)

    # Test datasets
    datasets = {
        'Validation': {
            'csv': OCRConfig.VAL_CSV,
            'folder': OCRConfig.VAL_DIR
        },
        'Test': {
            'csv': OCRConfig.TEST_CSV,
            'folder': OCRConfig.TEST_DIR
        }
    }

    # Evaluate on each dataset
    for dataset_name, paths in datasets.items():
        if os.path.exists(paths['csv']):
            print(f"\nEvaluating on {dataset_name} dataset...")

            # Evaluate medicine names
            results = evaluator.evaluate_dataset(
                paths['csv'],
                paths['folder'],
                target_column='MEDICINE_NAME'
            )

            # Print summary
            print(f"\n{dataset_name} Dataset Results (Medicine Names):")
            print(f"Average Edit Distance: {results['avg_edit_distance']:.4f}")
            print(f"Average Character Accuracy: {results['avg_char_accuracy']:.4f}")
            print(f"Average Word Accuracy: {results['avg_word_accuracy']:.4f}")

            # Generate reports
            report_path = f'{dataset_name.lower()}_evaluation_report.txt'
            plots_path = f'{dataset_name.lower()}_evaluation_plots.png'
            comparison_path = f'{dataset_name.lower()}_prediction_comparison.csv'

            evaluator.generate_evaluation_report(results, report_path)
            evaluator.plot_evaluation_metrics(results, plots_path)
            evaluator.compare_predictions(results, comparison_path)
            evaluator.analyze_common_errors(results)

        else:
            print(f"Dataset not found: {paths['csv']}")

    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
