from src.utils import get_collection_as_dataframe 
import pandas as pd
from src.logger import logging 
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split
from src.config import mongo_client
from src.utils import get_collection_as_dataframe
from src.exception import SrcException
import numpy as np 
import os, sys

# Data Ingestion class for handling data extraction and storage
class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f'{">"*20} Data Ingestion {"<"*20}')  # Log process start
            self.data_ingestion_config = data_ingestion_config  # Configuration settings
        except Exception as e:
            raise SrcException(e, sys)  # Handle exceptions

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info(f"Exporting data from MongoDB Atlas as a Pandas DataFrame")
            
            # Extract data from MongoDB
            df = get_collection_as_dataframe(
                mongo_client=mongo_client,
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name
            )
            
            logging.info(f"DataFrame contains Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            
           # Create feature store directory if it does not exist
            logging.info("Creating feature store folder if not available")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            
            # Save DataFrame to feature store
            df.to_csv(self.data_ingestion_config.feature_store_file_path, index=False, header=True)
            logging.info(f"Saved DataFrame to feature store folder")

            # Splitting dataset into training and testing sets
            logging.info(f"Splitting data into train and test sets using train_test_split")
            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.test_threshold)

            # Save train and test sets
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_file_path), exist_ok=True)
            train_df.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_file_path), exist_ok=True)
            test_df.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)
            logging.info(f"Saved train and test datasets")

            # Creating DataIngestionArtifact
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise SrcException(e, sys)
