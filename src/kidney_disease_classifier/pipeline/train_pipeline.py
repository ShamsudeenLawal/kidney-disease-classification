from src.kidney_disease_classifier.config.configuration import ConfigurationManager
from src.kidney_disease_classifier.components.data_ingestion import DataIngestion
from src.kidney_disease_classifier.components.prepare_base_model import PrepareBaseModel
from src.kidney_disease_classifier.components.model_training import Training
from src.kidney_disease_classifier.components.model_evaluation import Evaluation

class TrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run(self):
        # Data ingestion stage
        ingestion_config = self.config_manager.get_data_ingestion_config()
        ingestion = DataIngestion(ingestion_config)
        ingestion.run()
        # Model preparation stage
        model_preparation_config = self.config_manager.get_prepare_base_model_config()
        model_preparation = PrepareBaseModel(model_preparation_config)
        model_preparation.run()
        # Model training stage
        model_trainer_config = self.config_manager.get_training_config()
        model_trainer = Training(model_trainer_config)
        model_trainer.run()
        # Model evaluation stage
        model_evaluator_config = self.config_manager.get_evaluation_config()
        model_evaluator = Evaluation(model_evaluator_config)
        model_evaluator.run()

if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.run()
