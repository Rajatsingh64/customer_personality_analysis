from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig ,DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.logger import logging
from src.exception import SrcException
import os, sys


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
        data_ingestion_config = DataIngestionConfig(training_pipeline_config = training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config = data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed Successfully.")
        print("Data Ingestion Completed Successfully.")

        # Step 2: Data Validation
        logging.info("Starting Data Validation Process...")
        print("Starting Data Validation Process...")
        data_validation_config=DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact , data_validation_config=data_validation_config)
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("Data Validation Completed Successfully.")
        print("Data Data Validation Completed Successfully.")

    except Exception as e:
        raise SrcException(e, sys)  # Raise custom exception with system details
