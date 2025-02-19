from datetime import datetime
from src.logger import logging 
from src.exception import SrcException
import os, sys

class TrainingPipelineConfig:

    def __init__(self):
        
        try:
            # Create artifact directory with timestamp
            timestamp = datetime.now().strftime('%m%d%y__%H%M%S')
            self.artifact_directory = os.path.join(os.getcwd(), "artifact", timestamp)
            
            # Ensure artifact directory exists
            if not os.path.exists(self.artifact_directory):
                os.makedirs(self.artifact_directory, exist_ok=True)
            
            logging.info(f"Artifact directory created at: {self.artifact_directory}")
        
        except Exception as e:
           raise SrcException(e, sys)


# Data ingestion configuration
class DataIngestionConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
         
        self.database_name = "customer"    # MongoDB database name
        self.collection_name = "records"   # Collection name
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_directory, "data_ingestion")  # Data ingestion directory
        self.feature_store_file_path = os.path.join(self.data_ingestion_dir, "feature_store", "customer.csv")  # Original dataset
        self.train_file_path = os.path.join(self.data_ingestion_dir, 'datasets', "train.csv")  # Train dataset path
        self.test_file_path = os.path.join(self.data_ingestion_dir, "datasets", "test.csv")  # Test dataset path
        self.test_threshold = 0.3  # Test size threshold


class DataValidationConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.data_validation_dir=os.path.join(training_pipeline_config.artifact_directory , "data_validation")
        self.report_file_path =os.path.join(self.data_validation_dir , "report.yml")
        self.missing_threshold=0.3 
        self.base_dataset_file_path=os.path.join(os.getcwd(), "dataset/customer.csv")