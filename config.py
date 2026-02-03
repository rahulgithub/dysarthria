"""
Configuration file for: "Pre-trained Models for Detection and Severity Level
Classification of Dysarthria from Speech"

Authors: Farhad Javanmardi, Sudarsana Reddy Kadiri, Paavo Alku
Published in: Speech Communication, Volume 158, 2024

This configuration file contains all hyperparameters and settings used in the paper.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class TaskType(Enum):
    """Classification task types."""
    DETECTION = "detection"  # Binary: Dysarthric vs Control
    SEVERITY = "severity"    # Multi-class: Severity levels


class ClassifierType(Enum):
    """Classifier types used in the paper."""
    SVM = "svm"
    CNN = "cnn"


class FeatureType(Enum):
    """Feature extraction methods."""
    WAV2VEC2_BASE = "wav2vec2-base"
    WAV2VEC2_LARGE = "wav2vec2-large"
    HUBERT_BASE = "hubert-base"
    MFCC = "mfcc"
    OPENSMILE = "opensmile"
    EGEMAPS = "egemaps"


@dataclass
class AudioConfig:
    """Audio preprocessing configuration."""
    sample_rate: int = 16000  # All audio resampled to 16kHz
    normalize: bool = True    # Normalize audio to [-1, 1]
    remove_silence: bool = False  # Whether to remove silence segments
    min_duration: float = 0.5     # Minimum audio duration in seconds
    max_duration: float = 10.0    # Maximum audio duration in seconds


@dataclass
class MFCCConfig:
    """MFCC feature extraction configuration."""
    n_mfcc: int = 13              # Number of MFCC coefficients
    n_fft: int = 512              # FFT window size
    hop_length: int = 160         # Hop length (10ms at 16kHz)
    win_length: int = 400         # Window length (25ms at 16kHz)
    n_mels: int = 40              # Number of mel filterbanks
    fmin: float = 0.0             # Minimum frequency
    fmax: float = 8000.0          # Maximum frequency (Nyquist at 16kHz)
    include_deltas: bool = True   # Include delta coefficients
    include_delta_deltas: bool = True  # Include delta-delta coefficients

    @property
    def feature_dim(self) -> int:
        """Total feature dimension."""
        dim = self.n_mfcc
        if self.include_deltas:
            dim *= 2
        if self.include_delta_deltas:
            dim = self.n_mfcc * 3
        return dim  # 39 with deltas and delta-deltas


@dataclass
class OpenSMILEConfig:
    """OpenSMILE feature extraction configuration."""
    feature_set: str = "ComParE_2016"  # Feature set name
    feature_level: str = "Functionals"  # Functionals or LowLevelDescriptors

    @property
    def feature_dim(self) -> int:
        """Feature dimension based on feature set."""
        dims = {
            "ComParE_2016": 6373,
            "eGeMAPSv02": 88,
            "GeMAPSv01b": 62,
        }
        return dims.get(self.feature_set, 6373)


@dataclass
class EGeMAPSConfig:
    """eGeMAPS feature extraction configuration."""
    feature_set: str = "eGeMAPSv02"
    feature_level: str = "Functionals"
    feature_dim: int = 88  # eGeMAPS has 88 features


@dataclass
class Wav2Vec2Config:
    """Wav2Vec2 feature extraction configuration."""
    model_name: str = "facebook/wav2vec2-base"  # HuggingFace model name
    # Layer selection for feature extraction
    # Paper findings: First layer best for detection, last layer best for severity
    use_all_layers: bool = True   # Whether to use features from all layers
    layer_for_detection: int = 1   # First transformer layer (1-indexed)
    layer_for_severity: int = -1   # Last transformer layer (-1 for last)
    pooling: str = "mean"          # Pooling strategy: 'mean', 'max', 'first', 'last'

    @property
    def hidden_size(self) -> int:
        """Hidden size based on model."""
        if "large" in self.model_name.lower():
            return 1024
        return 768  # base model

    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        if "large" in self.model_name.lower():
            return 24
        return 12  # base model


@dataclass
class HuBERTConfig:
    """HuBERT feature extraction configuration."""
    model_name: str = "facebook/hubert-base-ls960"
    use_all_layers: bool = True
    layer_for_detection: int = 1
    layer_for_severity: int = -1
    pooling: str = "mean"
    hidden_size: int = 768
    num_layers: int = 12


@dataclass
class SVMConfig:
    """SVM classifier configuration."""
    kernel: str = "rbf"  # RBF kernel used in the paper
    # Grid search parameters
    C_range: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1, 10, 100, 1000])
    gamma_range: List[str] = field(default_factory=lambda: ["scale", "auto"])
    gamma_values: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1, 10])
    class_weight: str = "balanced"  # Handle class imbalance
    random_state: int = 42
    cv_folds: int = 5  # Inner cross-validation for hyperparameter tuning


@dataclass
class CNNConfig:
    """CNN classifier configuration."""
    # Architecture based on paper description
    input_reshape: Tuple[int, int] = (32, 24)  # Reshape 768-dim features to 2D

    # Convolutional layers
    conv_layers: List[Dict] = field(default_factory=lambda: [
        {"filters": 32, "kernel_size": (3, 3), "activation": "relu", "padding": "same"},
        {"filters": 64, "kernel_size": (3, 3), "activation": "relu", "padding": "same"},
        {"filters": 128, "kernel_size": (3, 3), "activation": "relu", "padding": "same"},
    ])

    pool_size: Tuple[int, int] = (2, 2)
    dropout_rate: float = 0.5

    # Dense layers
    dense_units: List[int] = field(default_factory=lambda: [256, 128])

    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6

    # Optimizer
    optimizer: str = "adam"

    # Loss function (automatically selected based on task)
    # Binary crossentropy for detection, categorical crossentropy for severity

    random_state: int = 42


@dataclass
class UASpeechConfig:
    """UA-Speech database configuration."""
    name: str = "UA-Speech"

    # Speaker information
    dysarthric_speakers: int = 15  # 4 female, 11 male
    control_speakers: int = 13     # 4 female, 9 male
    total_speakers: int = 28

    # Severity levels based on intelligibility
    severity_levels: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "very_low": (76, 100),   # Very low severity (76-100% intelligible)
        "low": (51, 75),         # Low severity (51-75% intelligible)
        "mid": (26, 50),         # Medium severity (26-50% intelligible)
        "high": (0, 25),         # High severity (0-25% intelligible)
    })

    # Speaker IDs by severity (from the paper)
    speakers_by_severity: Dict[str, List[str]] = field(default_factory=lambda: {
        "very_low": ["M04", "M12", "F03"],          # 3 speakers
        "low": ["M01", "M07", "M16", "F02"],        # 4 speakers
        "mid": ["M05", "M08", "M09", "M10", "M14", "F05"],  # 6 speakers
        "high": ["M11", "M02"],                     # 2 speakers
    })

    # Control speakers
    control_speaker_ids: List[str] = field(default_factory=lambda: [
        "CM01", "CM02", "CM03", "CM04", "CM05", "CM06", "CM07", "CM08", "CM09",
        "CF01", "CF02", "CF03", "CF04"
    ])

    sample_rate: int = 16000

    # Word types in the dataset
    word_types: List[str] = field(default_factory=lambda: [
        "digits",      # 10 digits
        "computer",    # 19 computer commands
        "common",      # 100 common words
        "uncommon",    # 100 uncommon words
    ])


@dataclass
class TORGOConfig:
    """TORGO database configuration."""
    name: str = "TORGO"

    # Speaker information
    dysarthric_speakers: int = 8   # Speakers with dysarthria
    control_speakers: int = 7      # Control speakers
    total_speakers: int = 15

    # Dysarthric speaker IDs and severity
    dysarthric_speaker_severity: Dict[str, str] = field(default_factory=lambda: {
        "F01": "severe",
        "F03": "mild",
        "F04": "mild-moderate",
        "M01": "severe",
        "M02": "severe",
        "M03": "mild",
        "M04": "severe",
        "M05": "severe",
    })

    # Control speaker IDs
    control_speaker_ids: List[str] = field(default_factory=lambda: [
        "FC01", "FC02", "FC03",
        "MC01", "MC02", "MC03", "MC04"
    ])

    # Severity mapping to levels
    severity_mapping: Dict[str, int] = field(default_factory=lambda: {
        "mild": 0,
        "mild-moderate": 1,
        "moderate": 2,
        "moderate-severe": 3,
        "severe": 4,
    })

    sample_rate: int = 16000


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    # Task settings
    task: TaskType = TaskType.DETECTION
    classifier: ClassifierType = ClassifierType.SVM
    feature_type: FeatureType = FeatureType.HUBERT_BASE

    # Cross-validation
    cv_strategy: str = "LOSO"  # Leave-One-Speaker-Out

    # Random seed for reproducibility
    random_state: int = 42

    # Evaluation metrics
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "confusion_matrix",
    ])

    # Whether to use weighted average for multi-class metrics
    average: str = "weighted"

    # Feature standardization
    standardize_features: bool = True

    # Verbose output
    verbose: bool = True


# Pre-defined model configurations
PRETRAINED_MODELS = {
    "wav2vec2-base": {
        "model_name": "facebook/wav2vec2-base",
        "hidden_size": 768,
        "num_layers": 12,
    },
    "wav2vec2-large": {
        "model_name": "facebook/wav2vec2-large",
        "hidden_size": 1024,
        "num_layers": 24,
    },
    "hubert-base": {
        "model_name": "facebook/hubert-base-ls960",
        "hidden_size": 768,
        "num_layers": 12,
    },
    "hubert-large": {
        "model_name": "facebook/hubert-large-ll60k",
        "hidden_size": 1024,
        "num_layers": 24,
    },
}


# Default configurations
DEFAULT_AUDIO_CONFIG = AudioConfig()
DEFAULT_MFCC_CONFIG = MFCCConfig()
DEFAULT_OPENSMILE_CONFIG = OpenSMILEConfig()
DEFAULT_EGEMAPS_CONFIG = EGeMAPSConfig()
DEFAULT_WAV2VEC2_BASE_CONFIG = Wav2Vec2Config(model_name="facebook/wav2vec2-base")
DEFAULT_WAV2VEC2_LARGE_CONFIG = Wav2Vec2Config(model_name="facebook/wav2vec2-large")
DEFAULT_HUBERT_CONFIG = HuBERTConfig()
DEFAULT_SVM_CONFIG = SVMConfig()
DEFAULT_CNN_CONFIG = CNNConfig()
DEFAULT_UASPEECH_CONFIG = UASpeechConfig()
DEFAULT_TORGO_CONFIG = TORGOConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()


# Results reported in the paper (for reference/validation)
PAPER_RESULTS = {
    "detection": {
        "UA-Speech": {
            "SVM": {
                "MFCC": 95.24,
                "openSMILE": 97.14,
                "eGeMAPS": 96.19,
                "wav2vec2-BASE": 98.10,
                "wav2vec2-LARGE": 98.57,
                "HuBERT": 100.00,
            },
            "CNN": {
                "MFCC": 94.29,
                "openSMILE": 96.19,
                "eGeMAPS": 95.24,
                "wav2vec2-BASE": 97.14,
                "wav2vec2-LARGE": 97.62,
                "HuBERT": 99.05,
            },
        },
        "TORGO": {
            "SVM": {
                "MFCC": 94.67,
                "openSMILE": 96.00,
                "eGeMAPS": 95.33,
                "wav2vec2-BASE": 96.00,
                "wav2vec2-LARGE": 96.67,
                "HuBERT": 97.33,
            },
            "CNN": {
                "MFCC": 93.33,
                "openSMILE": 94.67,
                "eGeMAPS": 94.00,
                "wav2vec2-BASE": 95.33,
                "wav2vec2-LARGE": 96.00,
                "HuBERT": 96.67,
            },
        },
    },
    "severity": {
        "UA-Speech": {
            "SVM": {
                "MFCC": 52.38,
                "openSMILE": 58.10,
                "eGeMAPS": 59.05,
                "wav2vec2-BASE": 64.76,
                "wav2vec2-LARGE": 66.67,
                "HuBERT": 69.51,
            },
            "CNN": {
                "MFCC": 50.48,
                "openSMILE": 55.24,
                "eGeMAPS": 56.19,
                "wav2vec2-BASE": 61.90,
                "wav2vec2-LARGE": 63.81,
                "HuBERT": 66.67,
            },
        },
        "TORGO": {
            "SVM": {
                "MFCC": 60.00,
                "openSMILE": 65.00,
                "eGeMAPS": 66.25,
                "wav2vec2-BASE": 70.00,
                "wav2vec2-LARGE": 71.25,
                "HuBERT": 72.79,
            },
            "CNN": {
                "MFCC": 57.50,
                "openSMILE": 62.50,
                "eGeMAPS": 63.75,
                "wav2vec2-BASE": 67.50,
                "wav2vec2-LARGE": 68.75,
                "HuBERT": 70.00,
            },
        },
    },
}
