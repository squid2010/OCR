# Medical Handwriting OCR System - Main CLI
import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import psutil

from config import OCRConfig
from data_loader import get_data_generators, load_single_image, load_batch_images, OCRDataLoader, build_vocab_from_labels
from ocr_model import build_ocr_model, decode_predictions, load_char_mappings
from predict import predict_medicine_name, batch_predict
from evaluate import evaluate_model

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class MemoryUsageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        mem_mb = mem_bytes / (1024 ** 2)
        print(f"\n[Memory] Epoch {epoch+1}: {mem_mb:.2f} MB used by process")

import gc
class GCGarbageCollector(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        print(f"[GC] Garbage collection forced at end of epoch {epoch+1}")

def ctc_loss_fn(y_true, y_pred):
    # Debug: print type and shape of y_true
    print("DEBUG ctc_loss_fn: y_true type:", type(y_true))
    print("DEBUG ctc_loss_fn: y_true shape:", getattr(y_true, "shape", None))
    # y_true: (batch, max_text_length)
    # y_pred: (batch, time_steps, num_classes)
    input_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    input_length = tf.expand_dims(input_length, axis=1)  # shape (batch_size, 1)
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, -1), tf.int32), axis=1)
    label_length = tf.expand_dims(label_length, axis=1)  # shape (batch_size, 1)
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def train(config):
    print("Starting training...")
    set_seed()
    # Data generators and loader lengths
    train_gen, val_gen, char_to_num, num_to_char = get_data_generators(config)
    vocab_size = len(char_to_num)
    print(f"Character vocab size: {vocab_size}")

    # Build loaders for length calculation
    train_csv = getattr(config, "TRAIN_LABELS", "trainingData/Training/training_labels.csv")
    val_csv = getattr(config, "VAL_LABELS", "trainingData/Validation/validation_labels.csv")
    train_img_dir = getattr(config, "TRAIN_DIR", "trainingData/Training/training_words")
    val_img_dir = getattr(config, "VAL_DIR", "trainingData/Validation/validation_words")
    img_height = getattr(config, "IMG_HEIGHT", 128)
    img_width = getattr(config, "IMG_WIDTH", 384)
    max_text_length = getattr(config, "MAX_TEXT_LENGTH", 32)
    batch_size = getattr(config, "BATCH_SIZE", 8)
    use_aug = getattr(config, "USE_AUGMENTATION", True)
    char_to_num, num_to_char, vocab = build_vocab_from_labels([train_csv, val_csv])

    train_loader = OCRDataLoader(
        csv_path=train_csv,
        images_dir=train_img_dir,
        char_to_num=char_to_num,
        img_height=img_height,
        img_width=img_width,
        max_text_length=max_text_length,
        batch_size=batch_size,
        augment=use_aug,
        shuffle=True,
    )
    val_loader = OCRDataLoader(
        csv_path=val_csv,
        images_dir=val_img_dir,
        char_to_num=char_to_num,
        img_height=img_height,
        img_width=img_width,
        max_text_length=max_text_length,
        batch_size=batch_size,
        augment=False,
        shuffle=False,
    )
    steps_per_epoch = len(train_loader)
    validation_steps = len(val_loader)

    # Build functional model (no Lambda layer)
    model, prediction_model = build_ocr_model(
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        num_chars=vocab_size,
        max_text_length=config.MAX_TEXT_LENGTH,
        lstm_units=config.LSTM_UNITS,
        dropout_rate=config.DROPOUT_RATE
    )
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/ocr_model_best.keras",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        MemoryUsageCallback(),  # Print memory usage after each epoch
        GCGarbageCollector(),   # Force garbage collection after each epoch
    ]

    # Mixed precision - DISABLED for memory safety
    tf.keras.mixed_precision.set_global_policy("float32")

    # Compile with custom CTC loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=ctc_loss_fn
    )

    # Fit
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_freq=5
    )

    # Save model for inference
    print("Saving model for inference to models/ocr_model_final.keras...")
    model.save("models/ocr_model_final.keras")
    print("Model for inference saved successfully.")
    # Save char mappings
    import pickle
    print("Saving character mappings to models/char_mappings.pkl...")
    with open("models/char_mappings.pkl", "wb") as f:
        pickle.dump({"char_to_num": char_to_num, "num_to_char": num_to_char}, f)
    print("Character mappings saved successfully.")
    # Save training history plot
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.legend()
        plt.title("Training History")
        plt.savefig("training_history.png")
        plt.close()
    except Exception as e:
        print("Could not save training history plot:", e)
    print("Training complete.")

    # Explicitly clear session and force GC
    tf.keras.backend.clear_session()
    del model, prediction_model, train_gen, val_gen, train_loader, val_loader
    import gc
    gc.collect()

class OCRModelEvaluator:
    """
    Comprehensive OCR Model Evaluation Class
    """

    def __init__(self, config=None):
        self.config = config if config else OCRConfig()
        self.char_to_num = None
        self.num_to_char = None
        self.model = None
        self.prediction_model = None
        self.results = {}

    def load_model_and_mappings(self, model_path=None, mappings_path=None):
        """Load the trained model and character mappings"""
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
                    # Try loading with custom objects
                    custom_objects = {'OCRModel': OCRModel}
                    self.prediction_model = tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)
                    print(f"✓ Successfully loaded prediction model from {path}")
                    return
                except Exception as e:
                    print(f"✗ Failed to load model from {path}: {e}")
                    # Try without custom objects as fallback
                    try:
                        self.prediction_model = tf.keras.models.load_model(path, compile=False)
                        print(f"✓ Successfully loaded prediction model from {path} (fallback)")
                        return
                    except Exception as e2:
                        print(f"✗ Fallback also failed: {e2}")
                        continue

        # Strategy 2: Try to load weights
        weight_files = [
            self.config.BEST_MODEL_PATH.replace('.h5', '.weights.h5'),
            self.config.FINAL_MODEL_PATH.replace('.h5', '.weights.h5'),
            self.config.BEST_MODEL_PATH,
            self.config.FINAL_MODEL_PATH
        ]

        print("Trying to create prediction model from weights...")
        for weights_path in weight_files:
            if os.path.exists(weights_path):
                try:
                    print(f"Attempting to load weights from {weights_path}")
                    # Build model first with dummy data
                    dummy_input = np.random.random((1, self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 1)).astype(np.float32)
                    _ = self.prediction_model(dummy_input)

                    # Load weights
                    self.prediction_model = tf.keras.models.load_model(path, compile=False)
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
        """Calculate character-level accuracy"""
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
        """Calculate word-level accuracy (exact match)"""
        correct_words = sum(1 for pred, gt in zip(predictions, ground_truth) if pred.strip() == gt.strip())
        return correct_words / len(predictions) if predictions else 0.0

    def calculate_edit_distance(self, predictions, ground_truth):
        """Calculate average edit distance (Levenshtein distance)"""
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
        """Calculate BLEU score for sequence evaluation"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
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
        except ImportError:
            print("NLTK not available, skipping BLEU score calculation")
            return 0.0

    def evaluate_dataset(self, dataset_type='validation', num_samples=None):
        """Evaluate model on specified dataset"""
        print(f"\nEvaluating on {dataset_type} dataset...")

        # Get dataset
        if dataset_type == 'validation':
            _, val_gen, _, _ = get_data_generators(self.config, only_val=True)
            dataset = val_gen
        else:
            # For test dataset, create similar generator
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
            'avg_predicted_length': np.mean(pred_lengths),
            'avg_ground_truth_length': np.mean(gt_lengths),
            'predictions': predictions[:50],  # Store first 50 for inspection
            'ground_truth': ground_truth[:50]
        }

        self.results[dataset_type] = results
        return results

    def print_evaluation_report(self, results):
        """Print comprehensive evaluation report"""
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
        """Save evaluation report to file"""
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
        """Create evaluation plots and visualizations"""
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
            edit_distances.append(self.calculate_edit_distance([pred], [gt]))

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

        if lengths and accuracies:
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
        """Save prediction comparison to CSV file"""
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

def evaluate(config, dataset='validation', num_samples=None, output_dir='evaluation_results', quick=False, demo=False):
    """Main evaluation function"""
    print("Starting OCR Model Evaluation")
    print(f"Dataset: {dataset}")
    print(f"Samples: {num_samples if num_samples else 'All'}")
    print(f"Output: {output_dir}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    # Set quick mode
    if quick:
        num_samples = 100
        print("Quick evaluation mode: Using 100 samples")

    if demo:
        print("Demo evaluation mode: Using simulated data")
        return evaluate_demo(output_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Initialize evaluator
        evaluator = OCRModelEvaluator(config)

        # Load model and mappings
        print("Loading model and character mappings...")
        evaluator.load_model_and_mappings()

        # Run evaluation
        print(f"Running evaluation on {dataset} dataset...")
        results = evaluator.evaluate_dataset(dataset, num_samples)

        # Print results
        evaluator.print_evaluation_report(results)

        # Save results
        print(f"\nSaving results to {output_dir}...")

        # Generate output filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"{dataset}_evaluation_report_{timestamp}.txt"
        plots_file = f"{dataset}_evaluation_plots_{timestamp}.png"
        csv_file = f"{dataset}_prediction_comparison_{timestamp}.csv"

        report_path = os.path.join(output_dir, report_file)
        plots_path = os.path.join(output_dir, plots_file)
        csv_path = os.path.join(output_dir, csv_file)

        # Save all results
        evaluator.save_evaluation_report(results, report_path)
        evaluator.create_evaluation_plots(results, plots_path)
        evaluator.save_prediction_comparison(results, csv_path)

        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to:")
        print(f"  - Report: {report_path}")
        print(f"  - Plots: {plots_path}")
        print(f"  - CSV: {csv_path}")

        # Print summary
        print(f"\nSUMMARY:")
        print(f"Character Accuracy: {results['character_accuracy']:.4f}")
        print(f"Word Accuracy: {results['word_accuracy']:.4f}")
        print(f"BLEU Score: {results['bleu_score']:.4f}")
        print(f"Average Edit Distance: {results['average_edit_distance']:.4f}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure your model and character mappings are properly saved.")
        print("Expected files:")
        print(f"  - {config.PREDICTION_MODEL_PATH}")
        print(f"  - {config.CHAR_MAPPINGS_PATH}")
        print("\nActual files found:")
        if os.path.exists(config.PREDICTION_MODEL_PATH):
            print(f"  ✓ {config.PREDICTION_MODEL_PATH}")
        else:
            print(f"  ✗ {config.PREDICTION_MODEL_PATH}")
        if os.path.exists(config.CHAR_MAPPINGS_PATH):
            print(f"  ✓ {config.CHAR_MAPPINGS_PATH}")
        else:
            print(f"  ✗ {config.CHAR_MAPPINGS_PATH}")
        print("\nTroubleshooting:")
        print("1. Check if the model file exists and has the correct extension (.keras or .h5)")
        print("2. Verify the model architecture matches the saved weights")
        print("3. If the model architecture changed, you may need to retrain the model")
        print("4. Run demo mode to test the evaluation framework: python3 demo_evaluation.py")
        sys.exit(1)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("\nIf you're having model loading issues, try:")
        print("1. Run demo mode: python3 demo_evaluation.py")
        print("2. Test model loading: python3 test_model_loading.py")
        print("3. Check compatibility: python3 check_model_compatibility.py")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def evaluate_demo(output_dir='demo_evaluation_results'):
    """Demo evaluation with simulated data"""
    print("=" * 60)
    print("OCR EVALUATION SYSTEM - DEMO MODE")
    print("=" * 60)
    print("This demo shows the evaluation framework using simulated OCR predictions.")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate demo data
    print("\nGenerating demo data...")
    np.random.seed(42)

    # Common medical terms for demo
    medical_terms = [
        "Paracetamol", "Aspirin", "Ibuprofen", "Omeprazole", "Metformin",
        "Amlodipine", "Simvastatin", "Lisinopril", "Atorvastatin", "Losartan",
        "Gabapentin", "Tramadol", "Codeine", "Morphine", "Diclofenac"
    ]

    # Generate ground truth
    ground_truth = []
    for i in range(100):
        term = np.random.choice(medical_terms)
        ground_truth.append(term)

    # Generate predictions with realistic errors
    predictions = []
    for gt in ground_truth:
        # 70% chance of perfect prediction
        if np.random.random() < 0.70:
            predictions.append(gt)
        else:
            # Generate different types of errors
            error_type = np.random.choice(['missing_char', 'extra_char', 'substitution'])

            if error_type == 'missing_char' and len(gt) > 3:
                pos = np.random.randint(1, len(gt) - 1)
                pred = gt[:pos] + gt[pos+1:]
            elif error_type == 'extra_char':
                pos = np.random.randint(0, len(gt))
                char = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'))
                pred = gt[:pos] + char + gt[pos:]
            elif error_type == 'substitution' and len(gt) > 0:
                pos = np.random.randint(0, len(gt))
                char = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'))
                pred = gt[:pos] + char + gt[pos+1:]
            else:
                pred = gt

            predictions.append(pred)

    print(f"Generated {len(predictions)} prediction samples")

    # Calculate metrics using OCRModelEvaluator methods
    evaluator = OCRModelEvaluator()

    char_accuracy = evaluator.calculate_character_accuracy(predictions, ground_truth)
    word_accuracy = evaluator.calculate_word_accuracy(predictions, ground_truth)
    avg_edit_distance = evaluator.calculate_edit_distance(predictions, ground_truth)
    bleu_score = evaluator.calculate_bleu_score(predictions, ground_truth)

    pred_lengths = [len(pred) for pred in predictions]
    gt_lengths = [len(gt) for gt in ground_truth]

    # Compile results
    results = {
        'dataset_type': 'demo',
        'num_samples': len(predictions),
        'character_accuracy': char_accuracy,
        'word_accuracy': word_accuracy,
        'average_edit_distance': avg_edit_distance,
        'bleu_score': bleu_score,
        'avg_predicted_length': np.mean(pred_lengths),
        'avg_ground_truth_length': np.mean(gt_lengths),
        'predictions': predictions[:50],
        'ground_truth': ground_truth[:50]
    }

    # Print results
    evaluator.print_evaluation_report(results)

    # Save results
    print(f"\nSaving demo results to {output_dir}...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"demo_evaluation_report_{timestamp}.txt"
    plots_file = f"demo_evaluation_plots_{timestamp}.png"
    csv_file = f"demo_prediction_comparison_{timestamp}.csv"

    report_path = os.path.join(output_dir, report_file)
    plots_path = os.path.join(output_dir, plots_file)
    csv_path = os.path.join(output_dir, csv_file)

    evaluator.save_evaluation_report(results, report_path)
    evaluator.create_evaluation_plots(results, plots_path)
    evaluator.save_prediction_comparison(results, csv_path)

    print(f"\nDemo evaluation completed successfully!")
    print(f"Results saved to '{output_dir}':")
    print(f"  - Report: {report_file}")
    print(f"  - Plots: {plots_file}")
    print(f"  - CSV: {csv_file}")

    print(f"\nNext steps:")
    print("1. This demo shows that the evaluation framework works correctly")
    print("2. To evaluate your real model, use: python3 run_ocr.py --evaluate")
    print("3. Make sure your model files are properly saved first")

def predict_cli(config, args):
    if args.image:
        # Single image prediction
        result = predict_medicine_name(args.image, visualize=args.visualize)
        print(f"Medicine: {result['medicine_name']}")
        print(f"Confidence: {result['confidence']:.2f}")
    elif args.folder:
        # Batch prediction
        batch_predict(
            args.folder,
            output_csv=args.output or "results.csv"
        )
        print(f"Batch prediction results saved to {args.output or 'results.csv'}")
    else:
        print("Please provide --image or --folder for prediction.")

def main():
    parser = argparse.ArgumentParser(
        description="Medical Handwriting OCR System"
    )
    parser.add_argument("--train", action="store_true", help="Train the OCR model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the OCR model")
    parser.add_argument("--predict", action="store_true", help="Predict using the OCR model")
    parser.add_argument("--image", type=str, help="Path to image for prediction")
    parser.add_argument("--folder", type=str, help="Path to folder for batch prediction")
    parser.add_argument("--output", type=str, help="Output CSV for batch prediction")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions")
    parser.add_argument("--config", type=str, default="OCRConfig", help="Config class to use")

    # Evaluation-specific arguments
    parser.add_argument("--dataset", type=str, default="validation",
                       choices=["validation", "test"], help="Dataset to evaluate")
    parser.add_argument("--samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick evaluation with 100 samples")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Output directory for evaluation results")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo evaluation with simulated data")

    args = parser.parse_args()

    # Select config
    config_module = __import__("config")
    config_class = getattr(config_module, args.config, OCRConfig)
    config = config_class()

    # Ensure output dirs
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    if args.train:
        train(config)
    elif args.evaluate:
        # --- Updated Evaluation Section --
        evaluate_model(num_samples=args.samples)
    elif args.predict:
        predict_cli(config, args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
