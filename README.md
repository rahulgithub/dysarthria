# Pre-trained Models for Detection and Severity Level Classification of Dysarthria from Speech

Implementation of the paper by Javanmardi, Kadiri, and Alku (Speech Communication, 2024).

## Paper Overview

### What is Dysarthria?

Dysarthria is a motor speech disorder caused by neurological damage (e.g., Cerebral Palsy, Parkinson's, stroke). It affects the muscles used for speech, resulting in:
- Slurred or slow speech
- Reduced intelligibility
- Abnormal pitch, loudness, or rhythm

**Clinical importance:** Early detection and severity assessment help clinicians make decisions about medication and therapy.

### Tasks Addressed

1. **Detection:** Is the speaker dysarthric or healthy? (Binary classification)
2. **Severity Classification:** How severe is the dysarthria? (4-class: Very Low, Low, Medium, High)

### Key Findings

| Finding | Details |
|---------|---------|
| Best features | HuBERT outperforms all baselines |
| Layer selection | First layer for detection, Last layer for severity |
| Best classifier | SVM slightly outperforms CNN |
| Detection accuracy | Up to 100% (HuBERT + SVM on UA-Speech) |
| Severity accuracy | Up to 69.51% (HuBERT + SVM on UA-Speech) |

---

## Installation

```bash
pip install torch torchaudio transformers librosa opensmile scikit-learn pandas numpy matplotlib seaborn tqdm
```

## Project Structure

```
├── config.py                                    # All configurations and hyperparameters
├── utils.py                                     # Feature extraction, models, evaluation
├── dysarthria_detection_severity_classification.ipynb  # Main notebook
└── README.md                                    # This file
```

---

## Methodology

### Datasets

| Dataset | Dysarthric Speakers | Control Speakers | Total |
|---------|---------------------|------------------|-------|
| **UA-Speech** | 15 (Cerebral Palsy) | 13 | 28 |
| **TORGO** | 8 (various causes) | 7 | 15 |

**Severity levels** (UA-Speech, based on intelligibility):
- Very Low severity: 76-100% intelligible
- Low severity: 51-75% intelligible
- Medium severity: 26-50% intelligible
- High severity: 0-25% intelligible

### Feature Extraction Methods

**Baseline Features:**
| Feature | Dimension | Description |
|---------|-----------|-------------|
| MFCC | 39 | 13 MFCCs + deltas + delta-deltas |
| openSMILE | 6,373 | ComParE 2016 feature set |
| eGeMAPS | 88 | Geneva Minimalistic Acoustic Parameter Set |

**Pre-trained Model Features:**
| Model | Dimension | Layers | Source |
|-------|-----------|--------|--------|
| wav2vec2-BASE | 768 | 12 | `facebook/wav2vec2-base` |
| wav2vec2-LARGE | 1024 | 24 | `facebook/wav2vec2-large` |
| HuBERT | 768 | 12 | `facebook/hubert-base-ls960` |

### Classifiers

1. **SVM** - RBF kernel with grid search for C and gamma
2. **CNN** - 3 convolutional layers + dense layers

### Evaluation

**Leave-One-Speaker-Out (LOSO) Cross-Validation:**
- Each fold: 1 speaker for testing, all others for training
- Ensures speaker-independent evaluation

---

## Step-by-Step Severity Classification Walkthrough

### Step 1: Define Severity Levels

Severity levels are defined in `config.py` based on speech intelligibility:

```python
# UA-Speech severity levels based on intelligibility percentage
severity_levels = {
    "very_low": (76, 100),   # 76-100% intelligible - least severe
    "low": (51, 75),         # 51-75% intelligible
    "mid": (26, 50),         # 26-50% intelligible
    "high": (0, 25),         # 0-25% intelligible - most severe
}

# Which speakers belong to which severity level
speakers_by_severity = {
    "very_low": ["M04", "M12", "F03"],                    # 3 speakers
    "low": ["M01", "M07", "M16", "F02"],                  # 4 speakers
    "mid": ["M05", "M08", "M09", "M10", "M14", "F05"],    # 6 speakers
    "high": ["M11", "M02"],                               # 2 speakers
}
```

### Step 2: Load Dataset & Assign Severity Labels

When loading the UA-Speech dataset for severity classification (`utils.py`):

```python
class UASpeechDataset:
    def _load_data(self):
        # For each speaker, look up their severity level
        for severity, speakers in self.config.speakers_by_severity.items():
            for speaker in speakers:
                # Load all audio files for this speaker
                for audio_file in speaker_dir.glob("*.wav"):
                    self.audio_files.append(str(audio_file))
                    self.labels.append(1)  # Dysarthric
                    self.speaker_ids.append(speaker_id)
                    self.severity_labels.append(severity)  # "very_low", "low", "mid", "high"

    def get_data(self, task):
        if task == TaskType.SEVERITY:
            # Filter to ONLY dysarthric speakers (exclude controls)
            dysarthric_mask = self.labels == 1

            # Convert string labels to numeric:
            severity_map = {"very_low": 0, "low": 1, "mid": 2, "high": 3}
            severity_labels = np.array([
                severity_map[s] for s in self.severity_labels if dysarthric
            ])

            return files, severity_labels, speakers
```

**Key point:** For severity classification, we ONLY use dysarthric speakers (15 speakers), not controls.

### Step 3: Extract Features

For severity classification, the paper found **last layer** features work best:

```python
# Audio file - Load & Preprocess
audio, sr = load_audio(file_path)  # Returns 16kHz normalized audio

# Extract features using HuBERT (or wav2vec2)
extractor = PretrainedFeatureExtractor("facebook/hubert-base-ls960")

features = extractor.extract_features(
    audio,
    sr=16000,
    layer=-1,       # LAST layer for severity (layer=1 for detection)
    pooling="mean"  # Average across time dimension
)
# Result: 768-dimensional feature vector per audio file
```

**The feature extraction flow:**

```
Audio Waveform (16kHz)
        |
        v
   HuBERT Model
        |
        v
   12 Transformer Layers
        |
        v
   Take LAST layer output: (batch, time_steps, 768)
        |
        v
   Mean pooling over time: (768,)
        |
        v
   Feature Vector for this utterance
```

### Step 4: Leave-One-Speaker-Out Cross-Validation

This is the core evaluation loop (`utils.py`):

```python
def leave_one_speaker_out_cv(features, labels, speaker_ids, classifier_type):
    logo = LeaveOneGroupOut()

    # For each unique speaker (15 dysarthric speakers)
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(features, labels, speaker_ids)):

        # SPLIT: Leave one speaker out for testing
        X_train = features[train_idx]   # Features from 14 speakers
        X_test = features[test_idx]     # Features from 1 held-out speaker
        y_train = labels[train_idx]     # Severity labels: 0,1,2,3
        y_test = labels[test_idx]

        # TRAIN classifier (SVM or CNN)
        if classifier_type == ClassifierType.SVM:
            classifier = SVMClassifier()
            classifier.fit(X_train, y_train)  # Grid search for C, gamma
        else:
            classifier = CNNClassifier(num_classes=4)  # 4 severity levels
            classifier.fit(X_train, y_train)

        # PREDICT on held-out speaker
        y_pred = classifier.predict(X_test)

        # EVALUATE this fold
        fold_accuracy = accuracy_score(y_test, y_pred)

    # Aggregate all predictions across all folds
    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
```

**Visual example of LOSO with 4 speakers:**

```
Fold 1: Train on [S02, S03, S04], Test on [S01]
Fold 2: Train on [S01, S03, S04], Test on [S02]
Fold 3: Train on [S01, S02, S04], Test on [S03]
Fold 4: Train on [S01, S02, S03], Test on [S04]
```

### Step 5: SVM Classifier Training

```python
class SVMClassifier:
    def fit(self, X, y):
        # 1. Standardize features (zero mean, unit variance)
        X_scaled = self.scaler.fit_transform(X)

        # 2. Grid search for best hyperparameters
        param_grid = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "gamma": [0.001, 0.01, 0.1, 1, 10, "scale", "auto"],
        }

        svm = SVC(kernel="rbf", class_weight="balanced")

        # 5-fold CV within training set to find best C, gamma
        grid_search = GridSearchCV(svm, param_grid, cv=5)
        grid_search.fit(X_scaled, y)

        self.model = grid_search.best_estimator_

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)  # Returns: [0, 2, 1, 3, ...]
```

### Step 6: CNN Classifier Training

```python
class CNNClassifierModel(nn.Module):
    def __init__(self, input_dim=768, num_classes=4):
        # Reshape 768-dim vector to 2D: (32, 24) for CNN processing

        # 3 Convolutional layers
        self.conv1 = Conv2d(1, 32, kernel_size=3)   # -> ReLU -> MaxPool
        self.conv2 = Conv2d(32, 64, kernel_size=3)  # -> ReLU -> MaxPool
        self.conv3 = Conv2d(64, 128, kernel_size=3) # -> ReLU -> MaxPool

        # Dense layers
        self.fc1 = Linear(flattened_size, 256)  # -> ReLU -> Dropout(0.5)
        self.fc2 = Linear(256, 128)             # -> ReLU -> Dropout(0.5)
        self.output = Linear(128, 4)            # 4 classes for severity

    def forward(self, x):
        # x shape: (batch, 768)
        x = x.view(batch, 1, 32, 24)  # Reshape to image-like

        # Conv layers with pooling
        x = pool(relu(conv1(x)))
        x = pool(relu(conv2(x)))
        x = pool(relu(conv3(x)))

        # Flatten and dense layers
        x = flatten(x)
        x = dropout(relu(fc1(x)))
        x = dropout(relu(fc2(x)))

        return self.output(x)  # Logits for 4 classes
```

Training uses **CrossEntropyLoss** (multi-class):

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)  # labels are 0,1,2,3
```

### Step 7: Final Evaluation

```python
def evaluate_classifier(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }

# Confusion matrix shows predictions vs actual:
#              Predicted
#            0    1    2    3
# Actual 0 [[10,  2,  0,  0],   # Very Low
#        1  [ 1, 15,  3,  0],   # Low
#        2  [ 0,  4, 20,  2],   # Mid
#        3  [ 0,  0,  1,  8]]   # High
```

### Complete Pipeline Flow

```
+------------------------------------------------------------------+
|  1. LOAD DATA                                                    |
|     - Load audio files for 15 dysarthric speakers                |
|     - Assign severity labels: 0=very_low, 1=low, 2=mid, 3=high   |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|  2. EXTRACT FEATURES (for each audio file)                       |
|     audio.wav -> HuBERT(last layer) -> mean pool -> 768-dim vec  |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|  3. LOSO CROSS-VALIDATION (15 folds, one per speaker)            |
|     For each fold:                                               |
|       - Train on 14 speakers                                     |
|       - Test on 1 held-out speaker                               |
|       - Record predictions                                       |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|  4. AGGREGATE RESULTS                                            |
|     - Combine all fold predictions                               |
|     - Calculate: Accuracy, Precision, Recall, F1                 |
|     - Generate confusion matrix                                  |
+------------------------------------------------------------------+
```

---

## Results from Paper

### Detection Task (Accuracy %)

| Feature | UA-Speech (SVM) | UA-Speech (CNN) | TORGO (SVM) | TORGO (CNN) |
|---------|-----------------|-----------------|-------------|-------------|
| MFCC | 95.24 | 94.29 | 94.67 | 93.33 |
| openSMILE | 97.14 | 96.19 | 96.00 | 94.67 |
| eGeMAPS | 96.19 | 95.24 | 95.33 | 94.00 |
| wav2vec2-BASE | 98.10 | 97.14 | 96.00 | 95.33 |
| wav2vec2-LARGE | 98.57 | 97.62 | 96.67 | 96.00 |
| **HuBERT** | **100.00** | **99.05** | **97.33** | **96.67** |

### Severity Classification (Accuracy %)

| Feature | UA-Speech (SVM) | UA-Speech (CNN) | TORGO (SVM) | TORGO (CNN) |
|---------|-----------------|-----------------|-------------|-------------|
| MFCC | 52.38 | 50.48 | 60.00 | 57.50 |
| openSMILE | 58.10 | 55.24 | 65.00 | 62.50 |
| eGeMAPS | 59.05 | 56.19 | 66.25 | 63.75 |
| wav2vec2-BASE | 64.76 | 61.90 | 70.00 | 67.50 |
| wav2vec2-LARGE | 66.67 | 63.81 | 71.25 | 68.75 |
| **HuBERT** | **69.51** | **66.67** | **72.79** | **70.00** |

---

## Usage

### Quick Start with Synthetic Data

```python
from utils import (
    create_synthetic_dataset,
    leave_one_speaker_out_cv,
    SVMClassifier,
)
from config import TaskType, ClassifierType

# Create synthetic data for testing
features, labels, speakers = create_synthetic_dataset(
    n_samples=300,
    n_features=768,
    n_speakers=15,
    task=TaskType.SEVERITY,
)

# Run LOSO cross-validation
results = leave_one_speaker_out_cv(
    features,
    labels,
    speakers,
    classifier_type=ClassifierType.SVM,
)

print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
```

### With Real Data

```python
from utils import UASpeechDataset, extract_features_batch, leave_one_speaker_out_cv
from config import TaskType, FeatureType, ClassifierType

# Load dataset
dataset = UASpeechDataset("./data/UA-Speech")
audio_files, labels, speakers = dataset.get_data(TaskType.SEVERITY)

# Extract HuBERT features (last layer for severity)
features = extract_features_batch(
    audio_files,
    feature_type=FeatureType.HUBERT_BASE,
    layer=-1,  # Last layer
)

# Evaluate with LOSO CV
results = leave_one_speaker_out_cv(
    features, labels, speakers,
    classifier_type=ClassifierType.SVM,
)
```

---

## References

1. Javanmardi, F., Kadiri, S. R., & Alku, P. (2024). Pre-trained models for detection and severity level classification of dysarthria from speech. *Speech Communication*, 158, 103047.

2. Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. *NeurIPS*.

3. Hsu, W. N., et al. (2021). HuBERT: Self-supervised speech representation learning by masked prediction of hidden units. *IEEE/ACM TASLP*.

4. Kim, H., et al. (2008). Dysarthric speech database for universal access research (UA-Speech). *Interspeech*.

5. Rudzicz, F., et al. (2012). The TORGO database of acoustic and articulatory speech from speakers with dysarthria. *Language Resources and Evaluation*.

---

## License

This implementation is for educational and research purposes.
