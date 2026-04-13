import os
from pathlib import Path
from kidney_disease_classifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from kidney_disease_classifier.utils.common import read_yaml, create_directories, save_json
from kidney_disease_classifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        # load/read project configuration and parameters from files (return is a ConfigBox object)
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # creating root artificat directory ("artifacts")
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        # collect data ingestion configurations
        config = self.config.data_ingestion
        # create data ingestion artifact root directory
        create_directories([config.root_dir])
        # instantiate data ingestion configuration object
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        # collect configuration for base model preparation
        config = self.config.prepare_base_model
        # create artifact root directory for base model preparation
        create_directories([config.root_dir])
        # instantiate a base model preparation config object
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_model_type=self.params.MODEL_TYPE,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:

        # collect training configurations
        training = self.config.training
        
        # collect base model preparation configurations
        prepare_base_model = self.config.prepare_base_model
        
        # instantiated parameters from class constructor
        params = self.params
        
        # specify training data path
        # training_data = os.path.join(self.config.data_ingestion.unzip_dir, "kidney-ct-scan-image")
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "raw")
        
        # create artifact root directory for training
        create_directories([
            Path(training.root_dir)
        ])
        
        # instantiate training configuration object
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        
        training = self.config.training
        data_ingestion = self.config.data_ingestion
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "raw")

        eval_config = EvaluationConfig(
            path_of_model = training.trained_model_path,
            training_data = training_data,
            mlflow_uri = "",
            all_params = self.params,
            params_image_size = self.params.IMAGE_SIZE,
            params_batch_size = self.params.BATCH_SIZE
        )
        return eval_config

