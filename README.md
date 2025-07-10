# Medical Handwriting OCR System

A deep learning-based Optical Character Recognition (OCR) system specifically designed to recognize handwritten medical text, particularly medicine names and generic drug names. This system uses a CNN-LSTM architecture with CTC (Connectionist Temporal Classification) loss for sequence-to-sequence learning.

## Features

- **CNN-LSTM Architecture**: Combines Convolutional Neural Networks for feature extraction and LSTM networks for sequence modeling
- **CTC Loss**: Handles variable-length sequences without requiring character-level alignment
- **Bidirectional LSTM**: Captures context from both directions for better accuracy
- **Data Augmentation**: Built-in augmentation to improve model robustness
- **Mixed Precision Training**: Faster training with reduced memory usage
- **Comprehensive Evaluation**: Multiple metrics including edit distance, character accuracy, and word accuracy
- **Batch Prediction**: Efficient processing of multiple images
- **Visualization Tools**: Built-in tools to visualize predictions and training progress

## Project Structure

```
OCR/
├── config.py              # Configuration settings
├── data_loader.py         # Data loading and preprocessing utilities
├── ocr_model.py          # Model architecture definition
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── predict.py            # Inference script
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── trainingData/         # Dataset directory
│   ├── Training/
│   │   ├── training_labels.csv
│   │   └── training_words
│   │        └── [image files]
│   ├── Validation/
│   │   ├── validation_labels.csv
│   │   └── validation_words
│   │        └── [image files]
│   └── Testing/
│       ├── testing_labels.csv
│       └── testing_words
│            └── [image files]
├── models/               # Saved models directory (created during training)
└── data                  # Saved data directory (created after evaluation)


## Installation

1. **Clone the repository** (if applicable) or ensure you have all the files in the OCR directory.

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify TensorFlow installation**:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Dataset Format

The dataset should be organized as follows:

- **Images**: PNG format, preferably grayscale
- **Labels**: CSV files with columns: `IMAGE`, `MEDICINE_NAME`, `GENERIC_NAME`

Example CSV format:
```csv
IMAGE,MEDICINE_NAME,GENERIC_NAME
0.png,Aceta,Paracetamol
1.png,Napa,Paracetamol
2.png,Axodin,Fexofenadine Hydrochloride
```

## Usage

### 1. Training the Model

To train the OCR model on your dataset:

```bash
python3 run_ocr.py --train
```

**Training Features**:
- Automatically creates character vocabulary from all labels
- Uses CTC loss for sequence learning
- Implements early stopping and learning rate reduction
- Saves best model weights and final model
- Generates training history plots

**Training Output**:
- `models/ocr_model_best.h5` - Best model weights
- `models/ocr_model_final.h5` - Final model weights
- `models/ocr_model_prediction.h5` - Model for inference
- `models/char_mappings.pkl` - Character to number mappings
- `training_history.png` - Training progress plots

### 2. Evaluating the Model

The evaluation system provides comprehensive metrics and analysis for your OCR model. You can evaluate using either the main evaluation script or the simple runner.

#### Using the Simple Evaluation Runner

```bash
# Quick evaluation with 100 samples
python3 run_evaluation.py --quick

# Full validation dataset evaluation
python3 run_evaluation.py --dataset validation

# Test dataset evaluation
python3 run_evaluation.py --dataset test --output results

# Evaluate specific number of samples
python3 run_evaluation.py --dataset validation --samples 500
```

#### Using the Main Evaluation Script

```bash
# Evaluate validation dataset
python3 evaluate.py --dataset validation

# Evaluate test dataset with specific number of samples
python3 evaluate.py --dataset test --num_samples 1000

# Custom model path
python3 evaluate.py --dataset validation --model_path models/custom_model.keras
```

**Evaluation Metrics**:
- **Character Accuracy**: Percentage of correctly predicted characters
- **Word Accuracy**: Percentage of completely correct word predictions
- **Edit Distance**: Average Levenshtein distance between predictions and ground truth
- **BLEU Score**: Sequence-level similarity score
- **CTC Loss**: Model's confidence in predictions
- **Length Statistics**: Analysis of predicted vs. actual text lengths

**Evaluation Features**:
- **Comprehensive Metrics**: Multiple evaluation metrics for thorough analysis
- **Visual Reports**: Automated generation of plots and charts
- **Error Analysis**: Detailed breakdown of prediction errors
- **Sample Inspection**: Display of sample predictions for manual review
- **Export Options**: Results saved in multiple formats (TXT, PNG, CSV)

**Evaluation Output**:
- `{dataset}_evaluation_report_{timestamp}.txt` - Detailed evaluation report
- `{dataset}_evaluation_plots_{timestamp}.png` - Visualization plots
- `{dataset}_prediction_comparison_{timestamp}.csv` - Detailed prediction comparisons

### 3. Making Predictions

#### Single Image Prediction

```bash
python3 run_ocr.py --predict --image path/to/image.png --visualize
```

#### Batch Prediction

```bash
python3 run_ocr.py --predict --folder path/to/images/ --output results.csv
```

#### Python API Usage

```python
from predict import predict_medicine_name

# Predict medicine name from image
result = predict_medicine_name('path/to/image.png')
print(f"Medicine: {result['medicine_name']}")
print(f"Confidence: {result['confidence']}")
```

## Model Architecture

The OCR model consists of several components:

1. **CNN Feature Extractor**:
   - 5 convolutional layers with increasing filters (32→64→128→128→256)
   - Max pooling for dimensionality reduction
   - Batch normalization for training stability
   - Dropout for regularization

2. **Sequence Modeling**:
   - Bidirectional LSTM layers (256 units each)
   - Handles variable-length sequences
   - Captures temporal dependencies

3. **CTC Output Layer**:
   - Dense layer with softmax activation
   - CTC loss for sequence alignment
   - Blank token for handling variable lengths

## Configuration

Model parameters can be modified in `config.py`:

```python
class OCRConfig:
    IMG_HEIGHT = 128
    IMG_WIDTH = 384
    MAX_TEXT_LENGTH = 32
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.001
    LSTM_UNITS = 256
    # ... more parameters
```

### Available Configurations

- **OCRConfig**: Default configuration
- **LightweightConfig**: Faster training, smaller model
- **HighAccuracyConfig**: Better accuracy, larger model
- **FastTrainingConfig**: Quick training for experimentation

## Performance Optimization

### Hardware Acceleration

- **GPU Support**: Automatically uses GPU if available
- **Mixed Precision**: Enabled by default for faster training
- **XLA Compilation**: Optional acceleration for compatible hardware

### Memory Optimization

- **Batch Processing**: Configurable batch sizes
- **Data Streaming**: Efficient data loading with tf.data
- **Model Checkpointing**: Saves only best weights

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**:
   ```bash
   # Check GPU availability
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

2. **Memory Errors**:
   - Reduce batch size in `config.py`
   - Disable mixed precision: `USE_MIXED_PRECISION = False`

3. **Training Not Converging**:
   - Check data quality and labels
   - Adjust learning rate
   - Increase training epochs

4. **Poor Predictions**:
   - Ensure images are preprocessed correctly
   - Check character mappings alignment
   - Verify model architecture parameters

### Performance Tips

1. **Data Quality**:
   - Ensure consistent image sizes
   - Remove corrupted images
   - Verify label accuracy

2. **Training**:
   - Use data augmentation for small datasets
   - Monitor validation loss for overfitting
   - Experiment with different learning rates

3. **Inference**:
   - Use batch prediction for multiple images
   - Consider model quantization for deployment
   - Cache character mappings for faster loading

## Model Metrics

The evaluation system provides comprehensive metrics to assess model performance:

### Primary Metrics

- **Character Accuracy**: Percentage of correctly predicted characters
  - Calculated by comparing each character position
  - Handles variable-length sequences with padding
  - Range: 0.0 to 1.0 (higher is better)

- **Word Accuracy**: Percentage of completely correct word predictions
  - Exact match between predicted and ground truth text
  - More stringent than character accuracy
  - Range: 0.0 to 1.0 (higher is better)

- **Edit Distance (Levenshtein Distance)**: Average character-level edit distance
  - Measures minimum operations needed to transform prediction to ground truth
  - Includes insertions, deletions, and substitutions
  - Range: 0 to max_text_length (lower is better)

- **BLEU Score**: Sequence-level similarity metric
  - Adapted from machine translation evaluation
  - Considers n-gram overlap between sequences
  - Range: 0.0 to 1.0 (higher is better)

- **CTC Loss**: Model's internal loss function value
  - Measures model confidence in predictions
  - Lower values indicate better model certainty
  - Range: 0.0 to infinity (lower is better)

### Additional Metrics

- **Length Statistics**: Analysis of text length patterns
  - Average predicted length vs. ground truth length
  - Helps identify systematic length biases
  - Useful for detecting over/under-segmentation

- **Accuracy by Length**: Performance breakdown by text length
  - Shows how accuracy varies with text complexity
  - Helps identify model limitations
  - Useful for targeted improvement

### Evaluation Reports

The evaluation system generates several types of reports:

1. **Text Report**: Comprehensive metrics summary
2. **Visual Plots**: Distribution charts and accuracy graphs
3. **CSV Export**: Detailed prediction comparisons for further analysis
4. **Sample Display**: Manual inspection of predictions vs. ground truth

### Interpreting Results

- **Character Accuracy > 0.90**: Good character-level recognition
- **Word Accuracy > 0.70**: Acceptable word-level performance
- **Edit Distance < 2.0**: Low error rate per word
- **BLEU Score > 0.8**: High sequence similarity
- **CTC Loss < 5.0**: Good model confidence

## Contributing

When contributing to this project:

1. Maintain consistent code style
2. Add unit tests for new features
3. Update documentation
4. Test on different datasets

## Future Improvements

Potential enhancements:

- **Attention Mechanism**: Add attention layers for better focus
- **Data Augmentation**: More sophisticated augmentation techniques
- **Multi-task Learning**: Simultaneously predict medicine and generic names
- **Model Ensemble**: Combine multiple models for better accuracy
- **Real-time Processing**: Optimize for real-time inference
- **Web Interface**: Add web-based prediction interface

## License

This project is intended for educational and research purposes. Please ensure compliance with relevant data protection and privacy regulations when using medical data.

## Citation

If you use this code in your research, please cite:

```
Medical Handwriting OCR System
Deep Learning-based OCR for Medical Text Recognition
```

---

For questions, issues, or contributions, please refer to the project maintainers.