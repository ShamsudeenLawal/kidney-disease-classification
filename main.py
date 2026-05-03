from src.kidney_disease_classifier import logger
from src.kidney_disease_classifier.pipeline.train_pipeline import TrainingPipeline

logger.info("Starting Training")
training_pipeline = TrainingPipeline()
training_pipeline.run()
logger.info("Training Completed")