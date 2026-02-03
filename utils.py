"""
Utility functions for: "Pre-trained Models for Detection and Severity Level
Classification of Dysarthria from Speech"

Authors: Farhad Javanmardi, Sudarsana Reddy Kadiri, Paavo Alku
Published in: Speech Communication, Volume 158, 2024

This module contains all utility functions for:
- Audio preprocessing
- Feature extraction (MFCC, openSMILE, eGeMAPS, wav2vec2, HuBERT)
- Data loading and preparation
- Model building (SVM, CNN)
- Training and evaluation
"""

import os
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Audio processing
import librosa
import soundfile as sf

# Machine Learning
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Configuration
from config import (
    AudioConfig,
    MFCCConfig,
    OpenSMILEConfig,
    EGeMAPSConfig,
    Wav2Vec2Config,
    HuBERTConfig,
    SVMConfig,
    CNNConfig,
    UASpeechConfig,
    TORGOConfig,
    ExperimentConfig,
    TaskType,
    ClassifierType,
    FeatureType,
    PRETRAINED_MODELS,
)

warnings.filterwarnings("ignore")


# =============================================================================
# Audio Loading and Preprocessing
# =============================================================================

def load_audio(
    file_path: str,
    config: AudioConfig = AudioConfig(),
) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess audio file.

    Args:
        file_path: Path to audio file
        config: Audio configuration

    Returns:
        Tuple of (audio signal, sample rate)
    """
    # Load audio
    audio, sr = librosa.load(file_path, sr=config.sample_rate)

    # Normalize
    if config.normalize:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

    # Trim silence if needed
    if config.remove_silence:
        audio, _ = librosa.effects.trim(audio, top_db=20)

    # Check duration constraints
    duration = len(audio) / sr
    if duration < config.min_duration:
        # Pad with zeros
        target_length = int(config.min_duration * sr)
        audio = np.pad(audio, (0, target_length - len(audio)))
    elif duration > config.max_duration:
        # Truncate
        target_length = int(config.max_duration * sr)
        audio = audio[:target_length]

    return audio, sr


def preprocess_audio_batch(
    file_paths: List[str],
    config: AudioConfig = AudioConfig(),
    show_progress: bool = True,
) -> List[Tuple[np.ndarray, int]]:
    """
    Load and preprocess multiple audio files.

    Args:
        file_paths: List of paths to audio files
        config: Audio configuration
        show_progress: Whether to show progress bar

    Returns:
        List of (audio, sample_rate) tuples
    """
    results = []
    iterator = tqdm(file_paths, desc="Loading audio") if show_progress else file_paths

    for path in iterator:
        try:
            audio, sr = load_audio(path, config)
            results.append((audio, sr))
        except Exception as e:
            print(f"Error loading {path}: {e}")
            results.append((None, None))

    return results


# =============================================================================
# MFCC Feature Extraction
# =============================================================================

def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    config: MFCCConfig = MFCCConfig(),
) -> np.ndarray:
    """
    Extract MFCC features from audio.

    Args:
        audio: Audio signal
        sr: Sample rate
        config: MFCC configuration

    Returns:
        MFCC features (time x features)
    """
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
    )

    features = [mfcc]

    # Add deltas
    if config.include_deltas:
        delta = librosa.feature.delta(mfcc)
        features.append(delta)

    # Add delta-deltas
    if config.include_delta_deltas:
        delta2 = librosa.feature.delta(mfcc, order=2)
        features.append(delta2)

    # Concatenate features
    features = np.concatenate(features, axis=0)

    # Transpose to (time, features)
    features = features.T

    return features


def extract_mfcc_stats(
    audio: np.ndarray,
    sr: int,
    config: MFCCConfig = MFCCConfig(),
) -> np.ndarray:
    """
    Extract MFCC features and compute statistics (mean, std).

    Args:
        audio: Audio signal
        sr: Sample rate
        config: MFCC configuration

    Returns:
        Flattened feature vector with statistics
    """
    mfcc = extract_mfcc(audio, sr, config)

    # Compute statistics over time
    mean = np.mean(mfcc, axis=0)
    std = np.std(mfcc, axis=0)

    # Concatenate mean and std
    features = np.concatenate([mean, std])

    return features


# =============================================================================
# openSMILE Feature Extraction
# =============================================================================

def extract_opensmile_features(
    audio: np.ndarray,
    sr: int,
    config: OpenSMILEConfig = OpenSMILEConfig(),
) -> np.ndarray:
    """
    Extract openSMILE features.

    Requires opensmile package: pip install opensmile

    Args:
        audio: Audio signal
        sr: Sample rate
        config: openSMILE configuration

    Returns:
        Feature vector
    """
    try:
        import opensmile
    except ImportError:
        raise ImportError("Please install opensmile: pip install opensmile")

    # Create smile extractor
    feature_set = getattr(opensmile.FeatureSet, config.feature_set)
    feature_level = getattr(opensmile.FeatureLevel, config.feature_level)

    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level,
    )

    # Extract features
    features = smile.process_signal(audio, sr)

    return features.values.flatten()


def extract_egemaps_features(
    audio: np.ndarray,
    sr: int,
    config: EGeMAPSConfig = EGeMAPSConfig(),
) -> np.ndarray:
    """
    Extract eGeMAPS features using openSMILE.

    Args:
        audio: Audio signal
        sr: Sample rate
        config: eGeMAPS configuration

    Returns:
        88-dimensional feature vector
    """
    try:
        import opensmile
    except ImportError:
        raise ImportError("Please install opensmile: pip install opensmile")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    features = smile.process_signal(audio, sr)

    return features.values.flatten()


# =============================================================================
# Pre-trained Model Feature Extraction (wav2vec2, HuBERT)
# =============================================================================

class PretrainedFeatureExtractor:
    """
    Feature extractor using pre-trained speech models (wav2vec2, HuBERT).
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        device: str = None,
    ):
        """
        Initialize feature extractor.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda' or 'cpu')
        """
        try:
            from transformers import (
                Wav2Vec2Model,
                Wav2Vec2Processor,
                HubertModel,
                AutoProcessor,
            )
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers"
            )

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and processor
        if "hubert" in model_name.lower():
            self.model = HubertModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            self.model = Wav2Vec2Model.from_pretrained(model_name)
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        # Get model config
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

    @torch.no_grad()
    def extract_features(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        layer: int = -1,
        pooling: str = "mean",
        return_all_layers: bool = False,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Extract features from audio using pre-trained model.

        Args:
            audio: Audio signal (should be at 16kHz)
            sr: Sample rate (should be 16000)
            layer: Which transformer layer to use (-1 for last, 0 for CNN output)
            pooling: Pooling strategy ('mean', 'max', 'first', 'last')
            return_all_layers: Whether to return features from all layers

        Returns:
            Feature vector or list of feature vectors (if return_all_layers)
        """
        # Ensure correct sample rate
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Process input
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs.input_values.to(self.device)

        # Forward pass with hidden states
        outputs = self.model(
            input_values,
            output_hidden_states=True,
        )

        # Get hidden states from all layers
        hidden_states = outputs.hidden_states  # Tuple of tensors

        if return_all_layers:
            # Return features from all layers
            all_features = []
            for hs in hidden_states:
                features = self._pool_features(hs, pooling)
                all_features.append(features.cpu().numpy())
            return all_features

        # Get features from specified layer
        if layer == -1:
            layer = len(hidden_states) - 1
        elif layer == 0:
            # CNN output (before transformer)
            layer = 0

        hidden_state = hidden_states[layer]
        features = self._pool_features(hidden_state, pooling)

        return features.cpu().numpy()

    def _pool_features(
        self,
        hidden_state: torch.Tensor,
        pooling: str,
    ) -> torch.Tensor:
        """
        Pool features across time dimension.

        Args:
            hidden_state: Hidden state tensor (batch, time, features)
            pooling: Pooling strategy

        Returns:
            Pooled features (batch, features)
        """
        if pooling == "mean":
            return torch.mean(hidden_state, dim=1).squeeze(0)
        elif pooling == "max":
            return torch.max(hidden_state, dim=1)[0].squeeze(0)
        elif pooling == "first":
            return hidden_state[:, 0, :].squeeze(0)
        elif pooling == "last":
            return hidden_state[:, -1, :].squeeze(0)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    def extract_layer_features(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        layer: int = 1,
        pooling: str = "mean",
    ) -> np.ndarray:
        """
        Extract features from a specific layer.

        According to the paper:
        - First layer features are best for detection
        - Last layer features are best for severity classification

        Args:
            audio: Audio signal
            sr: Sample rate
            layer: Layer number (1-indexed, 1 to num_layers)
            pooling: Pooling strategy

        Returns:
            Feature vector
        """
        return self.extract_features(audio, sr, layer=layer, pooling=pooling)


def extract_wav2vec2_base_features(
    audio: np.ndarray,
    sr: int = 16000,
    layer: int = -1,
    pooling: str = "mean",
    extractor: PretrainedFeatureExtractor = None,
) -> np.ndarray:
    """
    Extract wav2vec2-BASE features.

    Args:
        audio: Audio signal
        sr: Sample rate
        layer: Layer to use (-1 for last, 1 for first transformer layer)
        pooling: Pooling strategy
        extractor: Pre-initialized extractor (for efficiency)

    Returns:
        768-dimensional feature vector
    """
    if extractor is None:
        extractor = PretrainedFeatureExtractor("facebook/wav2vec2-base")

    return extractor.extract_features(audio, sr, layer=layer, pooling=pooling)


def extract_wav2vec2_large_features(
    audio: np.ndarray,
    sr: int = 16000,
    layer: int = -1,
    pooling: str = "mean",
    extractor: PretrainedFeatureExtractor = None,
) -> np.ndarray:
    """
    Extract wav2vec2-LARGE features.

    Args:
        audio: Audio signal
        sr: Sample rate
        layer: Layer to use
        pooling: Pooling strategy
        extractor: Pre-initialized extractor

    Returns:
        1024-dimensional feature vector
    """
    if extractor is None:
        extractor = PretrainedFeatureExtractor("facebook/wav2vec2-large")

    return extractor.extract_features(audio, sr, layer=layer, pooling=pooling)


def extract_hubert_features(
    audio: np.ndarray,
    sr: int = 16000,
    layer: int = -1,
    pooling: str = "mean",
    extractor: PretrainedFeatureExtractor = None,
) -> np.ndarray:
    """
    Extract HuBERT features.

    Args:
        audio: Audio signal
        sr: Sample rate
        layer: Layer to use
        pooling: Pooling strategy
        extractor: Pre-initialized extractor

    Returns:
        768-dimensional feature vector
    """
    if extractor is None:
        extractor = PretrainedFeatureExtractor("facebook/hubert-base-ls960")

    return extractor.extract_features(audio, sr, layer=layer, pooling=pooling)


# =============================================================================
# Feature Extraction Wrapper
# =============================================================================

def extract_features(
    audio: np.ndarray,
    sr: int,
    feature_type: FeatureType,
    layer: int = -1,
    pooling: str = "mean",
    extractor: PretrainedFeatureExtractor = None,
    **kwargs,
) -> np.ndarray:
    """
    Extract features using specified method.

    Args:
        audio: Audio signal
        sr: Sample rate
        feature_type: Type of features to extract
        layer: Layer for pretrained models
        pooling: Pooling strategy
        extractor: Pre-initialized extractor
        **kwargs: Additional arguments

    Returns:
        Feature vector
    """
    if feature_type == FeatureType.MFCC:
        return extract_mfcc_stats(audio, sr, MFCCConfig())

    elif feature_type == FeatureType.OPENSMILE:
        return extract_opensmile_features(audio, sr, OpenSMILEConfig())

    elif feature_type == FeatureType.EGEMAPS:
        return extract_egemaps_features(audio, sr, EGeMAPSConfig())

    elif feature_type == FeatureType.WAV2VEC2_BASE:
        return extract_wav2vec2_base_features(audio, sr, layer, pooling, extractor)

    elif feature_type == FeatureType.WAV2VEC2_LARGE:
        return extract_wav2vec2_large_features(audio, sr, layer, pooling, extractor)

    elif feature_type == FeatureType.HUBERT_BASE:
        return extract_hubert_features(audio, sr, layer, pooling, extractor)

    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def extract_features_batch(
    audio_files: List[str],
    feature_type: FeatureType,
    layer: int = -1,
    pooling: str = "mean",
    show_progress: bool = True,
) -> np.ndarray:
    """
    Extract features from multiple audio files.

    Args:
        audio_files: List of audio file paths
        feature_type: Type of features to extract
        layer: Layer for pretrained models
        pooling: Pooling strategy
        show_progress: Whether to show progress bar

    Returns:
        Feature matrix (n_samples, n_features)
    """
    # Initialize extractor for pretrained models
    extractor = None
    if feature_type in [
        FeatureType.WAV2VEC2_BASE,
        FeatureType.WAV2VEC2_LARGE,
        FeatureType.HUBERT_BASE,
    ]:
        model_name = PRETRAINED_MODELS[feature_type.value]["model_name"]
        extractor = PretrainedFeatureExtractor(model_name)

    features_list = []
    iterator = tqdm(audio_files, desc=f"Extracting {feature_type.value}") if show_progress else audio_files

    audio_config = AudioConfig()

    for file_path in iterator:
        try:
            audio, sr = load_audio(file_path, audio_config)
            features = extract_features(
                audio, sr, feature_type, layer, pooling, extractor
            )
            features_list.append(features)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            features_list.append(None)

    # Filter out failed extractions
    valid_features = [f for f in features_list if f is not None]

    return np.array(valid_features)


# =============================================================================
# Dataset Loading
# =============================================================================

class DysarthriaDataset:
    """
    Base class for dysarthria datasets.
    """

    def __init__(self, data_dir: str, config: Union[UASpeechConfig, TORGOConfig]):
        """
        Initialize dataset.

        Args:
            data_dir: Path to dataset directory
            config: Dataset configuration
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.audio_files = []
        self.labels = []
        self.speaker_ids = []
        self.severity_labels = []

    def load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata. Override in subclass."""
        raise NotImplementedError

    def get_data(
        self,
        task: TaskType = TaskType.DETECTION,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Get audio files, labels, and speaker IDs.

        Args:
            task: Task type (detection or severity)

        Returns:
            Tuple of (audio_files, labels, speaker_ids)
        """
        raise NotImplementedError


class UASpeechDataset(DysarthriaDataset):
    """
    UA-Speech dataset loader.

    Dataset structure (expected):
    data_dir/
        speaker_id/
            *.wav
    """

    def __init__(self, data_dir: str, config: UASpeechConfig = UASpeechConfig()):
        super().__init__(data_dir, config)
        self._load_data()

    def _load_data(self):
        """Load all audio files and metadata."""
        # Get all dysarthric speakers
        dysarthric_speakers = []
        for severity, speakers in self.config.speakers_by_severity.items():
            for speaker in speakers:
                dysarthric_speakers.append((speaker, severity))

        # Load dysarthric speaker files
        for speaker_id, severity in dysarthric_speakers:
            speaker_dir = self.data_dir / speaker_id
            if speaker_dir.exists():
                for audio_file in speaker_dir.glob("*.wav"):
                    self.audio_files.append(str(audio_file))
                    self.labels.append(1)  # Dysarthric
                    self.speaker_ids.append(speaker_id)
                    self.severity_labels.append(severity)

        # Load control speaker files
        for speaker_id in self.config.control_speaker_ids:
            speaker_dir = self.data_dir / speaker_id
            if speaker_dir.exists():
                for audio_file in speaker_dir.glob("*.wav"):
                    self.audio_files.append(str(audio_file))
                    self.labels.append(0)  # Control
                    self.speaker_ids.append(speaker_id)
                    self.severity_labels.append("control")

        self.labels = np.array(self.labels)
        self.speaker_ids = np.array(self.speaker_ids)

    def get_data(
        self,
        task: TaskType = TaskType.DETECTION,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Get data for specified task."""
        if task == TaskType.DETECTION:
            return self.audio_files, self.labels, self.speaker_ids
        else:
            # Filter to only dysarthric speakers for severity classification
            dysarthric_mask = self.labels == 1
            files = [f for f, m in zip(self.audio_files, dysarthric_mask) if m]
            speakers = self.speaker_ids[dysarthric_mask]

            # Map severity to numeric labels
            severity_map = {"very_low": 0, "low": 1, "mid": 2, "high": 3}
            severity_labels = np.array([
                severity_map[s] for s, m in zip(self.severity_labels, dysarthric_mask) if m
            ])

            return files, severity_labels, speakers


class TORGODataset(DysarthriaDataset):
    """
    TORGO dataset loader.

    Dataset structure (expected):
    data_dir/
        speaker_id/
            Session*/
                *.wav
    """

    def __init__(self, data_dir: str, config: TORGOConfig = TORGOConfig()):
        super().__init__(data_dir, config)
        self._load_data()

    def _load_data(self):
        """Load all audio files and metadata."""
        # Load dysarthric speaker files
        for speaker_id, severity in self.config.dysarthric_speaker_severity.items():
            speaker_dir = self.data_dir / speaker_id
            if speaker_dir.exists():
                for audio_file in speaker_dir.rglob("*.wav"):
                    self.audio_files.append(str(audio_file))
                    self.labels.append(1)  # Dysarthric
                    self.speaker_ids.append(speaker_id)
                    self.severity_labels.append(severity)

        # Load control speaker files
        for speaker_id in self.config.control_speaker_ids:
            speaker_dir = self.data_dir / speaker_id
            if speaker_dir.exists():
                for audio_file in speaker_dir.rglob("*.wav"):
                    self.audio_files.append(str(audio_file))
                    self.labels.append(0)  # Control
                    self.speaker_ids.append(speaker_id)
                    self.severity_labels.append("control")

        self.labels = np.array(self.labels)
        self.speaker_ids = np.array(self.speaker_ids)

    def get_data(
        self,
        task: TaskType = TaskType.DETECTION,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Get data for specified task."""
        if task == TaskType.DETECTION:
            return self.audio_files, self.labels, self.speaker_ids
        else:
            # Filter to only dysarthric speakers
            dysarthric_mask = self.labels == 1
            files = [f for f, m in zip(self.audio_files, dysarthric_mask) if m]
            speakers = self.speaker_ids[dysarthric_mask]

            # Map severity to numeric labels
            severity_labels = np.array([
                self.config.severity_mapping[s]
                for s, m in zip(self.severity_labels, dysarthric_mask) if m
            ])

            return files, severity_labels, speakers


def create_synthetic_dataset(
    n_samples: int = 100,
    n_features: int = 768,
    n_speakers: int = 10,
    task: TaskType = TaskType.DETECTION,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic dataset for testing.

    Args:
        n_samples: Number of samples
        n_features: Feature dimension
        n_speakers: Number of speakers
        task: Task type
        random_state: Random seed

    Returns:
        Tuple of (features, labels, speaker_ids)
    """
    np.random.seed(random_state)

    # Generate features
    features = np.random.randn(n_samples, n_features)

    # Generate speaker IDs
    speaker_ids = np.array([f"S{i % n_speakers:02d}" for i in range(n_samples)])

    if task == TaskType.DETECTION:
        # Binary labels
        labels = np.random.randint(0, 2, n_samples)
    else:
        # Multi-class severity labels (4 levels)
        labels = np.random.randint(0, 4, n_samples)

    return features, labels, speaker_ids


# =============================================================================
# SVM Classifier
# =============================================================================

class SVMClassifier:
    """
    SVM classifier with hyperparameter tuning via grid search.
    """

    def __init__(self, config: SVMConfig = SVMConfig()):
        """
        Initialize SVM classifier.

        Args:
            config: SVM configuration
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tune_hyperparameters: bool = True,
    ) -> "SVMClassifier":
        """
        Fit SVM classifier.

        Args:
            X: Feature matrix
            y: Labels
            tune_hyperparameters: Whether to tune hyperparameters

        Returns:
            Self
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        if tune_hyperparameters:
            # Grid search for hyperparameters
            param_grid = {
                "C": self.config.C_range,
                "gamma": self.config.gamma_values + self.config.gamma_range,
            }

            svm = SVC(
                kernel=self.config.kernel,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
            )

            grid_search = GridSearchCV(
                svm,
                param_grid,
                cv=self.config.cv_folds,
                scoring="accuracy",
                n_jobs=-1,
            )

            grid_search.fit(X_scaled, y)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
        else:
            self.model = SVC(
                kernel=self.config.kernel,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
            )
            self.model.fit(X_scaled, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Accuracy score
        """
        return accuracy_score(y, self.predict(X))


# =============================================================================
# CNN Classifier
# =============================================================================

class CNNClassifierModel(nn.Module):
    """
    CNN classifier model for dysarthria detection/classification.
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 2,
        config: CNNConfig = CNNConfig(),
    ):
        """
        Initialize CNN model.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            config: CNN configuration
        """
        super().__init__()

        self.config = config
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Compute reshape dimensions
        self.reshape_h = config.input_reshape[0]
        self.reshape_w = config.input_reshape[1]

        # Adjust if needed
        if self.reshape_h * self.reshape_w != input_dim:
            # Find closest factors
            for h in range(32, 0, -1):
                if input_dim % h == 0:
                    self.reshape_h = h
                    self.reshape_w = input_dim // h
                    break

        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        in_channels = 1
        current_h, current_w = self.reshape_h, self.reshape_w

        for conv_config in config.conv_layers:
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels,
                    conv_config["filters"],
                    kernel_size=conv_config["kernel_size"],
                    padding=1,
                )
            )
            self.pool_layers.append(
                nn.MaxPool2d(config.pool_size)
            )
            in_channels = conv_config["filters"]
            current_h = current_h // config.pool_size[0]
            current_w = current_w // config.pool_size[1]

        # Compute flattened size
        self.flat_size = in_channels * max(1, current_h) * max(1, current_w)

        # Build dense layers
        self.dense_layers = nn.ModuleList()
        prev_units = self.flat_size

        for units in config.dense_units:
            self.dense_layers.append(nn.Linear(prev_units, units))
            prev_units = units

        self.dropout = nn.Dropout(config.dropout_rate)
        self.output_layer = nn.Linear(prev_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, features)

        Returns:
            Output logits
        """
        # Reshape to 2D
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.reshape_h, self.reshape_w)

        # Convolutional layers
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = F.relu(conv(x))
            if x.size(2) > 1 and x.size(3) > 1:
                x = pool(x)

        # Flatten
        x = x.view(batch_size, -1)

        # Dense layers
        for dense in self.dense_layers:
            x = F.relu(dense(x))
            x = self.dropout(x)

        # Output
        x = self.output_layer(x)

        return x


class DysarthriaDatasetTorch(Dataset):
    """PyTorch Dataset for dysarthria classification."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CNNClassifier:
    """
    CNN classifier wrapper with training loop.
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 2,
        config: CNNConfig = CNNConfig(),
        device: str = None,
    ):
        """
        Initialize CNN classifier.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of classes
            config: CNN configuration
            device: Device to use
        """
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.model = CNNClassifierModel(input_dim, num_classes, config)
        self.model.to(self.device)

        self.scaler = StandardScaler()
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: bool = True,
    ) -> "CNNClassifier":
        """
        Train CNN classifier.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            verbose: Print progress

        Returns:
            Self
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Create dataset and dataloader
        train_dataset = DysarthriaDatasetTorch(X_scaled, y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_dataset = DysarthriaDatasetTorch(X_val_scaled, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_lr,
        )

        # Early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        # Training loop
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels).sum().item()
                train_total += labels.size(0)

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validation
            val_loss = 0
            val_acc = 0
            if val_loader is not None:
                self.model.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for features, labels in val_loader:
                        features = features.to(self.device)
                        labels = labels.to(self.device)

                        outputs = self.model(features)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_correct += predicted.eq(labels).sum().item()
                        val_total += labels.size(0)

                val_loss /= len(val_loader)
                val_acc = val_correct / val_total

                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.config.epochs} - "
                msg += f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                if val_loader is not None:
                    msg += f" - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                print(msg)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = outputs.max(1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Class probabilities
        """
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = F.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Accuracy score
        """
        return accuracy_score(y, self.predict(X))


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Evaluate classifier performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multi-class

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    return metrics


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
) -> pd.DataFrame:
    """
    Get confusion matrix as DataFrame.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels

    Returns:
        Confusion matrix DataFrame
    """
    cm = confusion_matrix(y_true, y_pred)

    if labels is None:
        labels = [str(i) for i in range(len(cm))]

    return pd.DataFrame(cm, index=labels, columns=labels)


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = None,
):
    """
    Print classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Class names
    """
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))


# =============================================================================
# Leave-One-Speaker-Out Cross-Validation
# =============================================================================

def leave_one_speaker_out_cv(
    features: np.ndarray,
    labels: np.ndarray,
    speaker_ids: np.ndarray,
    classifier_type: ClassifierType = ClassifierType.SVM,
    classifier_config: Union[SVMConfig, CNNConfig] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Perform Leave-One-Speaker-Out cross-validation.

    Args:
        features: Feature matrix
        labels: Labels
        speaker_ids: Speaker identifiers
        classifier_type: Type of classifier
        classifier_config: Classifier configuration
        verbose: Print progress

    Returns:
        Dictionary with results
    """
    logo = LeaveOneGroupOut()

    all_predictions = []
    all_true_labels = []
    all_speakers = []
    fold_accuracies = []

    unique_speakers = np.unique(speaker_ids)
    num_folds = len(unique_speakers)

    if verbose:
        print(f"Running LOSO CV with {num_folds} folds...")

    for fold_idx, (train_idx, test_idx) in enumerate(
        logo.split(features, labels, speaker_ids)
    ):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        test_speaker = speaker_ids[test_idx][0]

        # Initialize classifier
        if classifier_type == ClassifierType.SVM:
            config = classifier_config or SVMConfig()
            classifier = SVMClassifier(config)
            classifier.fit(X_train, y_train, tune_hyperparameters=True)
        else:
            config = classifier_config or CNNConfig()
            num_classes = len(np.unique(labels))
            classifier = CNNClassifier(
                input_dim=features.shape[1],
                num_classes=num_classes,
                config=config,
            )
            classifier.fit(X_train, y_train, verbose=False)

        # Predict
        y_pred = classifier.predict(X_test)

        # Store results
        all_predictions.extend(y_pred)
        all_true_labels.extend(y_test)
        all_speakers.extend([test_speaker] * len(y_test))

        fold_acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(fold_acc)

        if verbose:
            print(f"Fold {fold_idx + 1}/{num_folds} (Speaker {test_speaker}): Accuracy = {fold_acc:.4f}")

    # Compute overall metrics
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    metrics = evaluate_classifier(all_true_labels, all_predictions)
    metrics["fold_accuracies"] = fold_accuracies
    metrics["mean_fold_accuracy"] = np.mean(fold_accuracies)
    metrics["std_fold_accuracy"] = np.std(fold_accuracies)

    results = {
        "predictions": all_predictions,
        "true_labels": all_true_labels,
        "speakers": all_speakers,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(all_true_labels, all_predictions),
    }

    if verbose:
        print(f"\nOverall Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Mean Fold Accuracy: {metrics['mean_fold_accuracy']:.4f} ± {metrics['std_fold_accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")

    return results


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiment(
    features: np.ndarray,
    labels: np.ndarray,
    speaker_ids: np.ndarray,
    task: TaskType = TaskType.DETECTION,
    classifier_type: ClassifierType = ClassifierType.SVM,
    feature_name: str = "features",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a complete experiment.

    Args:
        features: Feature matrix
        labels: Labels
        speaker_ids: Speaker identifiers
        task: Task type (detection or severity)
        classifier_type: Classifier type
        feature_name: Name of features (for logging)
        verbose: Print progress

    Returns:
        Experiment results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {feature_name}")
        print(f"Task: {task.value}")
        print(f"Classifier: {classifier_type.value}")
        print(f"Samples: {len(labels)}, Features: {features.shape[1]}")
        print(f"{'='*60}")

    # Get classifier config
    if classifier_type == ClassifierType.SVM:
        config = SVMConfig()
    else:
        config = CNNConfig()

    # Run LOSO CV
    results = leave_one_speaker_out_cv(
        features,
        labels,
        speaker_ids,
        classifier_type,
        config,
        verbose,
    )

    results["task"] = task.value
    results["classifier"] = classifier_type.value
    results["feature_name"] = feature_name

    return results


def run_all_experiments(
    audio_files: List[str],
    labels: np.ndarray,
    speaker_ids: np.ndarray,
    task: TaskType = TaskType.DETECTION,
    feature_types: List[FeatureType] = None,
    classifier_types: List[ClassifierType] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run experiments with all feature types and classifiers.

    Args:
        audio_files: List of audio file paths
        labels: Labels
        speaker_ids: Speaker identifiers
        task: Task type
        feature_types: List of feature types to use
        classifier_types: List of classifier types to use
        verbose: Print progress

    Returns:
        DataFrame with all results
    """
    if feature_types is None:
        feature_types = list(FeatureType)

    if classifier_types is None:
        classifier_types = list(ClassifierType)

    results_list = []

    for feature_type in feature_types:
        if verbose:
            print(f"\nExtracting {feature_type.value} features...")

        # Determine layer based on task
        layer = 1 if task == TaskType.DETECTION else -1

        # Extract features
        features = extract_features_batch(
            audio_files,
            feature_type,
            layer=layer,
            show_progress=verbose,
        )

        for classifier_type in classifier_types:
            # Run experiment
            exp_results = run_experiment(
                features,
                labels,
                speaker_ids,
                task,
                classifier_type,
                feature_type.value,
                verbose,
            )

            results_list.append({
                "feature_type": feature_type.value,
                "classifier": classifier_type.value,
                "task": task.value,
                "accuracy": exp_results["metrics"]["accuracy"],
                "precision": exp_results["metrics"]["precision"],
                "recall": exp_results["metrics"]["recall"],
                "f1_score": exp_results["metrics"]["f1_score"],
                "mean_fold_acc": exp_results["metrics"]["mean_fold_accuracy"],
                "std_fold_acc": exp_results["metrics"]["std_fold_accuracy"],
            })

    return pd.DataFrame(results_list)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        labels: Class labels
        title: Plot title
        figsize: Figure size
        cmap: Colormap
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_results_comparison(
    results_df: pd.DataFrame,
    metric: str = "accuracy",
    title: str = "Results Comparison",
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Plot comparison of results.

    Args:
        results_df: DataFrame with results
        metric: Metric to plot
        title: Plot title
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # Pivot for grouped bar chart
    pivot_df = results_df.pivot(
        index="feature_type",
        columns="classifier",
        values=metric,
    )

    pivot_df.plot(kind="bar", ax=ax)

    ax.set_xlabel("Feature Type")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title)
    ax.legend(title="Classifier")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 4),
):
    """
    Plot CNN training history.

    Args:
        history: Training history dictionary
        title: Plot title
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss
    axes[0].plot(history["train_loss"], label="Train")
    if "val_loss" in history and len(history["val_loss"]) > 0:
        axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train")
    if "val_acc" in history and len(history["val_acc"]) > 0:
        axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.suptitle(title)
    plt.tight_layout()
    return fig


# =============================================================================
# Utility Functions
# =============================================================================

def save_results(
    results: Dict[str, Any],
    output_path: str,
):
    """
    Save experiment results.

    Args:
        results: Results dictionary
        output_path: Output file path
    """
    import json

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    serializable_results = convert_to_serializable(results)

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)


def load_results(input_path: str) -> Dict[str, Any]:
    """
    Load experiment results.

    Args:
        input_path: Input file path

    Returns:
        Results dictionary
    """
    import json

    with open(input_path, "r") as f:
        return json.load(f)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
