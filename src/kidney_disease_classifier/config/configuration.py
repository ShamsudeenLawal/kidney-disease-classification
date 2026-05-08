from pathlib import Path

from kidney_disease_classifier.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH
)

from kidney_disease_classifier.utils.common import (
    read_yaml,
    create_directories
)

from kidney_disease_classifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    # =========================================================
    # DATA INGESTION CONFIG
    # =========================================================
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([
            Path(config.root_dir),
            Path(config.raw_dir),
            Path(config.split.root_dir),
        ])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_url=config.source_url,
            local_data_file=Path(config.local_data_file),
            raw_dir=Path(config.raw_dir),
            split=config.split
        )

    # =========================================================
    # PREPARE BASE MODEL CONFIG
    # =========================================================
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        model_params = self.params.model
        training_params = self.params.training
        transfer_learning_params = self.params.transfer_learning

        create_directories([Path(config.root_dir)])

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),

            model_type=model_params.name,
            input_shape=model_params.input_shape,
            learning_rate=training_params.learning_rate,
            
            freeze_all=transfer_learning_params.freeze_all,
            freeze_till=transfer_learning_params.freeze_till,
            weights=model_params.weights,
            class_labels_path = Path(config.class_labels_path),
            train_dir = Path(self.config.data_ingestion.split.train_dir)
            
        )

    # =========================================================
    # TRAINING CONFIG
    # =========================================================
    def get_training_config(self) -> TrainingConfig:
        cfg = self.config
        params = self.params

        create_directories([Path(cfg.training.root_dir)])

        return TrainingConfig(
            root_dir=Path(cfg.training.root_dir),
            trained_model_path=Path(cfg.training.trained_model_path),
            updated_base_model_path=Path(
                cfg.prepare_base_model.updated_base_model_path
            ),

            train_dir=Path(cfg.training.data.train_dir),
            val_dir=Path(cfg.training.data.val_dir),

            model_name=params.model.name,
            input_shape=params.model.input_shape,
            # num_classes=params.model.num_classes,
            weights=params.model.weights,

            batch_size=params.training.batch_size,
            epochs=params.training.epochs,
            learning_rate=params.training.learning_rate,
            optimizer=params.training.optimizer,

            augmentation=params.augmentation,
            callbacks=params.training.callbacks,
            transfer_learning=params.transfer_learning,

            experiment_name=params.mlflow.experiment_name,
            tracking_uri=cfg.mlflow.tracking_uri,
            run_id_file=Path(cfg.mlflow.run_id_file),
        )

    # =========================================================
    # EVALUATION CONFIG
    # =========================================================
    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation
        training = self.config.training
        params = self.params.model
        mlflow_cfg = self.config.mlflow

        create_directories([Path(config.root_dir)])

        return EvaluationConfig(
            # -------- Paths --------
            model_path=Path(training.trained_model_path),
            test_dir=Path(config.data.test_dir),
            scores_path=Path(config.scores_path),

            # -------- Model Params (needed for preprocessing) --------
            model_name=params.name,
            input_shape=params.input_shape,
            batch_size=self.params.training.batch_size,

            # -------- Evaluation Metrics --------
            metrics=self.params.evaluation.metrics,

            # -------- MLflow --------
            tracking_uri=mlflow_cfg.tracking_uri,
            run_id_file=Path(mlflow_cfg.run_id_file),
        )