from src.entity.config_entity import (
    TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, 
    DataTransformationConfig, ModelTrainingConfig , ModelEvaluationConfig , ModelPusherConfig
)
from src.entity.artifact_entity import (
    DataIngestionArtifact, DataValidationArtifact, ModelTrainingArtifact ,ModelEvaluationArtifact , ModelPusherArtifact
)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.exception import SrcException
import os
import sys


def run_training_pipeline():      
    """
    Runs the entire training pipeline, starting with data ingestion.
    """
    try:  
        logging.info(f"{'>'*20} Running Training Pipeline {'<'*20}")
        print(f"{'>'*20} Running Training Pipeline {'<'*20}")
        
        # Initialize Training Pipeline Configuration
        training_pipeline_config = TrainingPipelineConfig() 
        
        # Step 1: Data Ingestion
        logging.info("Starting Data Ingestion Process...")
        print("Starting Data Ingestion Process...")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed Successfully.")
        print("Data Ingestion Completed Successfully.")

        # Step 2: Data Validation
        logging.info("Starting Data Validation Process...")
        print("Starting Data Validation Process...")
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact, 
            data_validation_config=data_validation_config
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Completed Successfully.")
        print("Data Validation Completed Successfully.")
        
        # Step 3: Data Transformation
        logging.info("Starting Data Transformation Process...")
        print("Starting Data Transformation Process...")
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(
            data_transformation_config=data_transformation_config, 
            data_ingestion_artifact=data_ingestion_artifact
        )
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation Completed Successfully.")
        print("Data Transformation Completed Successfully.")
        
        # Step 4: Model Training
        logging.info("Starting Model Training Process...")
        print("Starting Model Training Process...")
        model_training_config = ModelTrainingConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(
            model_trainer_config=model_training_config,
            data_ingestion_artifact=data_ingestion_artifact,
            data_transformation_artifact=data_transformation_artifact
        )
        model_training_artifact = model_trainer.initiate_model_training()  # Added missing method call
        logging.info("Model Training Completed Successfully.")
        print("Model Training Completed Successfully.")

        # Step 5: Model Evaluation 
        logging.info("Starting Model Evaluation Process...")
        print("Starting Model Evaluation Process...")
        model_evaluation_config=ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
        model_evaluation=ModelEvaluation(model_evaluation_config=model_evaluation_config ,
                                          model_trainer_artifact=model_training_artifact , 
                                          data_ingestion_artifact=data_ingestion_artifact ,
                                          data_transformation_artifact=data_transformation_artifact)
        model_evaluation_artifact=model_evaluation.initiate_model_evaluation()
        logging.info("Model Evaluation Completed Successfully.")
        print("Model Evaluation Completed Successfully.")
      
        # Step 6: Model Pusher 
        logging.info("Starting Model Pusher Process...")
        print("Starting Model Pusher Process...")
        model_pusher_config=ModelPusherConfig(training_pipeline_config=training_pipeline_config)
        model_pusher=ModelPusher(model_pusher_config=model_pusher_config , 
                                 data_transformation_artifact=data_transformation_artifact , 
                                 model_trainer_artifact=model_training_artifact)
        model_pusher_artifact=model_pusher.initiate_model_pusher()
        logging.info("Model Pusher Completed Successfully.")
        print("Model Pusher Completed Successfully.")
    
    except Exception as e:
        raise SrcException(e, sys)
