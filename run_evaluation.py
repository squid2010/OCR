#!/usr/bin/env python3
# run_evaluation.py
# Simple script to run OCR model evaluation

import os
import sys
import argparse
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import OCRConfig
from evaluate import OCRModelEvaluator

def main():
    """
    Simple evaluation runner with basic options
    """
    parser = argparse.ArgumentParser(description='Run OCR Model Evaluation')
    parser.add_argument('--dataset', type=str, default='validation',
                       choices=['validation', 'test'],
                       help='Dataset to evaluate (default: validation)')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick evaluation with 100 samples')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for results')

    args = parser.parse_args()

    import os
    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Set quick mode
    if args.quick:
        args.samples = 100
        print("Quick evaluation mode: Using 100 samples")

    print(f"Starting OCR Model Evaluation")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.samples if args.samples else 'All'}")
    print(f"Output: {args.output}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    try:
        # Initialize evaluator
        config = OCRConfig()
        evaluator = OCRModelEvaluator(config)

        # Load model and mappings
        print("Loading model and character mappings...")
        evaluator.load_model_and_mappings()

        # Run evaluation
        print(f"Running evaluation on {args.dataset} dataset...")
        results = evaluator.evaluate_dataset(args.dataset, args.samples)

        # Print results
        evaluator.print_evaluation_report(results)

        # Save results
        print(f"\nSaving results to {args.output}...")

        # Generate output filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"{args.dataset}_evaluation_report_{timestamp}.txt"
        plots_file = f"{args.dataset}_evaluation_plots_{timestamp}.png"
        csv_file = f"{args.dataset}_prediction_comparison_{timestamp}.csv"

        report_path = os.path.join(args.output, report_file)
        plots_path = os.path.join(args.output, plots_file)
        csv_path = os.path.join(args.output, csv_file)

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
        import os
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
        print("3. Try running the compatibility checker: python3 check_model_compatibility.py")
        print("4. If the model architecture changed, you may need to retrain the model")
        print("5. Run demo mode to test the evaluation framework: python3 demo_evaluation.py")
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

if __name__ == "__main__":
    main()
