# Medical Handwriting OCR System - Main CLI
import argparse
import os
import sys
import tensorflow as tf
import numpy as np

from config import OCRConfig
from data_loader import get_data_generators, load_single_image, load_batch_images
from ocr_model import build_ocr_model, decode_predictions, load_char_mappings
from predict import predict_medicine_name, batch_predict

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def train(config):
    print("Starting training...")
    set_seed()
    # Data generators and loader lengths
    from data_loader import get_data_generators, OCRDataLoader, build_vocab_from_labels
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

    # Model
    model, _ = build_ocr_model(
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
            filepath="models/ocr_model_best.weights.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
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
        )
    ]

    # Mixed precision
    if getattr(config, "USE_MIXED_PRECISION", True):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    )

    # Fit
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    # Save final weights
    model.save_weights("models/ocr_model_final.weights.h5")
    # Save model for inference
    model.save("models/ocr_model_prediction.keras")
    # Save char mappings
    import pickle
    with open("models/char_mappings.pkl", "wb") as f:
        pickle.dump({"char_to_num": char_to_num, "num_to_char": num_to_char}, f)
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

def evaluate(config):
    print("Starting evaluation...")
    set_seed()
    # Load char mappings
    char_to_num, num_to_char = load_char_mappings("models/char_mappings.pkl")
    vocab_size = len(char_to_num)
    # Load model
    model, _ = build_ocr_model(
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        num_chars=vocab_size,
        max_text_length=config.MAX_TEXT_LENGTH,
        lstm_units=config.LSTM_UNITS,
        dropout_rate=config.DROPOUT_RATE
    )
    model.load_weights("models/ocr_model_best.weights.h5")
    # Data generators
    _, val_gen, _, _ = get_data_generators(config, only_val=True)
    # Evaluate on validation
    from evaluate import evaluate_model
    val_report, val_plots, val_pred_csv = evaluate_model(
        model, val_gen, num_to_char, "validation"
    )
    print(f"Validation report saved to {val_report}")
    # Evaluate on test
    test_report, test_plots, test_pred_csv = evaluate_model(
        model, None, num_to_char, "test"
    )
    print(f"Test report saved to {test_report}")
    print("Evaluation complete.")

def predict_cli(config, args):
    if args.image:
        # Single image prediction
        result = predict_medicine_name(args.image, visualize=args.visualize)
        print(f"Medicine: {result['medicine_name']}")
        print(f"Confidence: {result['confidence']:.2f}")
    elif args.folder:
        # Batch prediction
        batch_predict(
            folder=args.folder,
            output_csv=args.output or "results.csv",
            visualize=args.visualize
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
        evaluate(config)
    elif args.predict:
        predict_cli(config, args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
