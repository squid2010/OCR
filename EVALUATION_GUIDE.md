# OCR Model Evaluation Guide

This guide provides comprehensive documentation for evaluating your OCR model using the built-in evaluation system.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Usage Examples](#usage-examples)
5. [Understanding Results](#understanding-results)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Overview

The OCR evaluation system provides comprehensive analysis of your model's performance using multiple metrics and visualization tools. It supports both validation and test dataset evaluation with detailed reporting capabilities.

### Key Features

- **Multiple Metrics**: Character accuracy, word accuracy, edit distance, BLEU score, and CTC loss
- **Visual Reports**: Automated generation of charts and distribution plots
- **Export Options**: Results saved in TXT, PNG, and CSV formats
- **Batch Processing**: Efficient evaluation of large datasets
- **Error Analysis**: Detailed breakdown of prediction errors
- **Sample Inspection**: Manual review of predictions vs ground truth

## Quick Start

### 1. Basic Evaluation

```bash
# Quick evaluation with 100 samples
python3 run_evaluation.py --quick

# Full validation dataset evaluation
python3 run_evaluation.py --dataset validation
```

### 2. Prerequisites

Ensure you have:
- A trained OCR model (saved as `.keras` or `.h5` file)
- Character mappings file (`char_mappings.pkl`)
- Test/validation dataset in the correct format

### 3. Expected File Structure

```
OCR/
├── models/
│   ├── ocr_model_prediction.keras    # Trained model
│   └── char_mappings.pkl             # Character mappings
├── trainingData/
│   ├── Validation/
│   │   ├── validation_labels.csv
│   │   └── validation_words/
│   └── Testing/
│       ├── testing_labels.csv
│       └── testing_words/
└── evaluation_results/               # Output directory
```

## Evaluation Metrics

### Primary Metrics

#### 1. Character Accuracy
- **Definition**: Percentage of correctly predicted characters
- **Calculation**: Compares each character position between prediction and ground truth
- **Range**: 0.0 to 1.0 (higher is better)
- **Good Performance**: > 0.90

```python
# Example calculation
prediction = "hello"
ground_truth = "helo"
# Character accuracy = 4/5 = 0.80
```

#### 2. Word Accuracy
- **Definition**: Percentage of completely correct word predictions
- **Calculation**: Exact match between predicted and ground truth text
- **Range**: 0.0 to 1.0 (higher is better)
- **Good Performance**: > 0.70

```python
# Example calculation
predictions = ["hello", "world", "test"]
ground_truth = ["hello", "word", "testing"]
# Word accuracy = 1/3 = 0.33 (only "hello" matches exactly)
```

#### 3. Edit Distance (Levenshtein Distance)
- **Definition**: Average number of character operations needed to transform prediction to ground truth
- **Operations**: Insertions, deletions, substitutions
- **Range**: 0 to max_text_length (lower is better)
- **Good Performance**: < 2.0

```python
# Example calculation
prediction = "hello"
ground_truth = "helo"
# Edit distance = 1 (one insertion of 'l')
```

#### 4. BLEU Score
- **Definition**: Sequence-level similarity metric adapted from machine translation
- **Calculation**: Considers n-gram overlap between sequences
- **Range**: 0.0 to 1.0 (higher is better)
- **Good Performance**: > 0.8

#### 5. CTC Loss
- **Definition**: Model's internal loss function value
- **Interpretation**: Measures model confidence in predictions
- **Range**: 0.0 to infinity (lower is better)
- **Good Performance**: < 5.0

### Secondary Metrics

#### Length Statistics
- **Average Predicted Length**: Mean length of predicted text
- **Average Ground Truth Length**: Mean length of actual text
- **Length Bias**: Difference between predicted and actual lengths

#### Accuracy by Length
- **Purpose**: Shows how performance varies with text complexity
- **Usage**: Identifies model limitations for short/long texts

## Usage Examples

### Basic Evaluation Commands

```bash
# Evaluate validation dataset
python3 run_evaluation.py --dataset validation

# Evaluate test dataset
python3 run_evaluation.py --dataset test

# Evaluate with custom sample size
python3 run_evaluation.py --dataset validation --samples 500

# Save results to custom directory
python3 run_evaluation.py --dataset validation --output my_results
```

### Advanced Evaluation Commands

```bash
# Using the main evaluation script
python3 evaluate.py --dataset validation --num_samples 1000

# Custom model path
python3 evaluate.py --dataset test --model_path models/my_model.keras

# Specific output directory
python3 evaluate.py --dataset validation --output_dir results
```

### Programmatic Usage

```python
from config import OCRConfig
from evaluate import OCRModelEvaluator

# Initialize evaluator
config = OCRConfig()
evaluator = OCRModelEvaluator(config)

# Load model and mappings
evaluator.load_model_and_mappings()

# Run evaluation
results = evaluator.evaluate_dataset('validation', num_samples=100)

# Print results
evaluator.print_evaluation_report(results)

# Save results
evaluator.save_evaluation_report(results, 'my_report.txt')
evaluator.create_evaluation_plots(results, 'my_plots.png')
evaluator.save_prediction_comparison(results, 'my_comparison.csv')
```

## Understanding Results

### Console Output

```
OCR MODEL EVALUATION REPORT - VALIDATION
============================================================
Dataset: validation
Number of samples: 1000
Evaluation date: 2024-01-15 14:30:22

Metrics                        Value          
---------------------------------------------
Character Accuracy             0.8750         
Word Accuracy                  0.7200         
Average Edit Distance          1.5000         
BLEU Score                     0.8100         
CTC Loss                       3.2000         
Avg Predicted Length           8.50           
Avg Ground Truth Length        8.00           

Sample Predictions (First 10)
------------------------------------------------------------
✓ Pred: 'Paracetamol' | GT: 'Paracetamol'
✗ Pred: 'Asprin' | GT: 'Aspirin'
✓ Pred: 'Ibuprofen' | GT: 'Ibuprofen'
...
```

### Generated Files

#### 1. Evaluation Report (`*_evaluation_report_*.txt`)
- Detailed metrics summary
- Sample predictions with ground truth
- Performance analysis

#### 2. Evaluation Plots (`*_evaluation_plots_*.png`)
- **Metrics Bar Chart**: Visual comparison of accuracy metrics
- **Length Distribution**: Histogram of predicted vs ground truth lengths
- **Edit Distance Distribution**: Frequency of edit distances
- **Accuracy by Length**: Performance variation by text length

#### 3. Prediction Comparison (`*_prediction_comparison_*.csv`)
- Complete prediction dataset
- Columns: Ground_Truth, Prediction, Match, Edit_Distance, Lengths
- Suitable for further analysis in Excel/Python

### Interpreting Performance

#### Excellent Performance
- Character Accuracy: > 0.95
- Word Accuracy: > 0.85
- Edit Distance: < 1.0
- BLEU Score: > 0.9

#### Good Performance
- Character Accuracy: 0.90-0.95
- Word Accuracy: 0.70-0.85
- Edit Distance: 1.0-2.0
- BLEU Score: 0.8-0.9

#### Needs Improvement
- Character Accuracy: < 0.90
- Word Accuracy: < 0.70
- Edit Distance: > 2.0
- BLEU Score: < 0.8

## Advanced Usage

### Custom Evaluation Metrics

```python
class CustomEvaluator(OCRModelEvaluator):
    def calculate_medical_accuracy(self, predictions, ground_truth):
        """Custom metric for medical text accuracy"""
        # Implement custom logic
        pass
    
    def evaluate_by_medicine_type(self, predictions, ground_truth):
        """Evaluate performance by medicine category"""
        # Implement category-specific evaluation
        pass
```

### Batch Evaluation

```python
# Evaluate multiple models
models = ['model_v1.keras', 'model_v2.keras', 'model_v3.keras']
results = {}

for model_path in models:
    evaluator = OCRModelEvaluator()
    evaluator.load_model_and_mappings(model_path)
    results[model_path] = evaluator.evaluate_dataset('validation')

# Compare results
for model, result in results.items():
    print(f"{model}: Word Accuracy = {result['word_accuracy']:.3f}")
```

### Integration with Training

```python
# Evaluate during training
def evaluate_model_callback(model, epoch):
    evaluator = OCRModelEvaluator()
    evaluator.prediction_model = model
    results = evaluator.evaluate_dataset('validation', num_samples=100)
    
    # Log metrics
    print(f"Epoch {epoch}: Word Accuracy = {results['word_accuracy']:.3f}")
    
    return results['word_accuracy']
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```bash
Error: Model not found at models/ocr_model_prediction.keras
```
**Solution**: Ensure model file exists and path is correct

#### 2. Character Mapping Errors
```bash
Error: Character mappings not found at models/char_mappings.pkl
```
**Solution**: Verify character mappings file exists and is compatible

#### 3. Dataset Errors
```bash
Error: No such file or directory: 'trainingData/Validation/validation_labels.csv'
```
**Solution**: Check dataset paths in config.py

#### 4. Memory Issues
```bash
Error: Resource exhausted: OOM when allocating tensor
```
**Solution**: Reduce batch size or use smaller sample size

### Performance Issues

#### Slow Evaluation
- Use `--quick` flag for fast evaluation
- Reduce sample size with `--samples N`
- Ensure GPU is being utilized

#### Large Memory Usage
- Reduce batch size in config
- Use smaller sample sizes
- Close other applications

### Debugging Tips

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test with minimal data
evaluator.evaluate_dataset('validation', num_samples=10)

# Check model output shapes
dummy_input = np.random.random((1, 128, 384, 1))
output = evaluator.prediction_model(dummy_input)
print(f"Model output shape: {output.shape}")
```

## Best Practices

### 1. Regular Evaluation
- Evaluate after each training epoch
- Use validation set for model selection
- Reserve test set for final evaluation

### 2. Sample Size Selection
- Use 100-500 samples for quick checks
- Use full dataset for final evaluation
- Balance speed vs accuracy needs

### 3. Metric Interpretation
- Focus on word accuracy for practical applications
- Use character accuracy for detailed analysis
- Monitor edit distance for error patterns

### 4. Result Storage
- Save evaluation results with timestamps
- Store results in version control
- Document model versions and configurations

### 5. Comparative Analysis
- Compare multiple model versions
- Track performance over time
- Identify improvement trends

### 6. Error Analysis
- Review sample predictions manually
- Identify common error patterns
- Use insights for model improvement

## Example Workflow

```bash
# 1. Quick check during development
python3 run_evaluation.py --quick

# 2. Detailed validation evaluation
python3 run_evaluation.py --dataset validation --output validation_results

# 3. Final test evaluation
python3 run_evaluation.py --dataset test --output final_results

# 4. Compare results
python3 -c "
import pandas as pd
val_df = pd.read_csv('validation_results/validation_prediction_comparison_*.csv')
test_df = pd.read_csv('final_results/test_prediction_comparison_*.csv')
print(f'Validation accuracy: {val_df.Match.mean():.3f}')
print(f'Test accuracy: {test_df.Match.mean():.3f}')
"
```

This comprehensive evaluation system will help you assess your OCR model's performance thoroughly and make data-driven decisions for model improvements.