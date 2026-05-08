from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    raw_dir: Path
    split: dict

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path

    model_type: str
    input_shape: list
    learning_rate: float

    freeze_all: bool
    freeze_till: int
    weights: str
    class_labels_path: Path
    train_dir: Path

@dataclass(frozen=True)
class TrainingConfig:
    # ---------------- ARTIFACT PATHS ----------------
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path

    # ---------------- DATA ----------------
    train_dir: Path
    val_dir: Path

    # ---------------- MODEL ----------------
    model_name: str
    input_shape: list
    # num_classes: int
    weights: str

    # ---------------- TRAINING ----------------
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str

    # ---------------- AUGMENTATION ----------------
    augmentation: Dict[str, Any]   # full augmentation config

    # ---------------- CALLBACKS ----------------
    callbacks: Dict[str, Any]

    # ---------------- TRANSFER LEARNING ----------------
    transfer_learning: Dict[str, Any]

    # ---------------- MLFLOW ----------------
    experiment_name: str
    tracking_uri: str
    run_id_file: Path

@dataclass(frozen=True)
class EvaluationConfig:
    model_path: Path

    test_dir: Path

    metrics_output_path: Path

    mlflow_uri: str
    experiment_name: str
    run_id_file: Path

    model_type: str
    image_size: list
    batch_size: int


from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class EvaluationConfig:
    
    model_path: Path
    test_dir: Path
    scores_path: Path

    model_name: str
    input_shape: List[int]
    batch_size: int

    metrics: List[str]

    tracking_uri: str
    run_id_file: Path