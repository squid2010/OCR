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
│   │   └── [image files]
│   ├── Validation/
│   │   ├── validation_labels.csv
│   │   └── [image files]
│   └── Testing/
│       ├── testing_labels.csv
│       └── [image files]
└── models/               # Saved models directory (created during training)
```

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
python train.py
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

To evaluate the trained model on test/validation data:

```bash
python evaluate.py
```

**Evaluation Features**:
- Tests on both validation and test datasets
- Calculates multiple metrics:
  - Edit distance
  - Character-level accuracy
  - Word-level accuracy
- Generates detailed reports and visualizations
- Analyzes common prediction errors

**Evaluation Output**:
- `validation_evaluation_report.txt` - Detailed validation results
- `test_evaluation_report.txt` - Detailed test results
- `*_evaluation_plots.png` - Metric distribution plots
- `*_prediction_comparison.csv` - Detailed prediction comparisons

### 3. Making Predictions

#### Single Image Prediction

```bash
python predict.py --image path/to/image.png --visualize
```

#### Batch Prediction

```bash
python predict.py --folder path/to/images/ --output results.csv
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

The system provides comprehensive evaluation metrics:

- **Edit Distance**: Levenshtein distance between predicted and actual text
- **Character Accuracy**: Percentage of correctly predicted characters
- **Word Accuracy**: Percentage of completely correct predictions
- **Confidence Score**: Model confidence in predictions

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