# Emotion Detection

Emotion Detection is a multi-modal deep learning project designed to analyze emotions in images. It predicts both categorical (e.g., happiness, sadness) and continuous (Valence, Arousal, Dominance) emotion attributes using a combination of contextual and body features.

---

## Features

- Multi-modal architecture combining context and body image features.
- Predicts:
  - **Categorical emotions** (26 categories such as happiness, anger, sadness).
  - **Continuous emotions** (Valence, Arousal, Dominance).
- Training, evaluation, and inference pipelines.
- Pre-trained model weights for efficient inference.
- Evaluation metrics include Average Precision (AP) and VAD errors.
- Plots precision-recall curves and determines thresholds.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shourya102/emotion-detection.git
   cd emotion-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the following files in the `models` directory:
   - Pre-trained weights file: `best_emotic_model.pth`
   - Thresholds file: `thresholds.npy`

---

## Usage

### Training

Train the model using the provided `train.py` script:
```bash
python models/train.py
```

### Evaluation

Evaluate the model on a test dataset using the `test.py` script:
```bash
python models/test.py
```

This script generates:
- Average Precision (AP) scores for each category.
- VAD (Valence, Arousal, Dominance) errors.
- Precision-recall curves.

### Inference

Run inference on an image to predict emotions:
```bash
python models/inference.py
```

Example usage in code:
```python
from models.inference import get_infer

image_path = "path/to/image.jpg"
predicted_emotions, continuous_values = get_infer(image_path)

print("Predicted Emotions:", predicted_emotions)
print("Continuous Values:", continuous_values)
```

---

## Architecture

### Model: MultiModalEmotic

- **Context Network**: ResNet-50 for extracting contextual features.
- **Body Network**: ResNet-18 for extracting body-specific features.
- **Fusion Layer**: Combines features from both networks.
- **Output Heads**:
  - Continuous Head: Predicts Valence, Arousal, and Dominance.
  - Categorical Head: Predicts 26 emotion categories.

### Dataset

The dataset consists of:
- **Annotations**: CSV files for train, validation, and test splits.
- **Images**:
  - Contextual images.
  - Body images (cropped regions around detected faces).

---

## Evaluation Metrics

1. **Categorical Emotions**:
   - Average Precision (AP) per category.
   - Mean Average Precision (mAP) across all categories.

2. **Continuous Emotions**:
   - Mean Absolute Error (MAE) for Valence, Arousal, and Dominance.

---

## Acknowledgements

- **Libraries Used**: PyTorch, torchvision, OpenCV, NumPy, Matplotlib, Pandas.
- **Pre-trained Models**: ResNet-50 and ResNet-18 from torchvision.
- **Dataset**: EMOTIC

This project provides a robust foundation for emotion detection using multi-modal deep learning techniques.
