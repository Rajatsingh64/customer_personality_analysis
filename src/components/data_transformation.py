import os
import sys
import dill
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing import Optional
from src.entity import config_entity, artifact_entity
from src.entity.config_entity import DataValidationConfig, DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.exception import SrcException
from src.logger import logging
from src.config import features_to_drop, outliers_handling_features
from src.utils import save_object, load_object, save_numpy_array_data, outliers_threshold, handling_outliers, handle_num_correlations
from src import utils
import warnings

warnings.filterwarnings("ignore")


class DataTransformation:
    def __init__(self, 
                 data_transformation_config: DataTransformationConfig, 
                 data_ingestion_artifact: DataIngestionArtifact):
        """
        Initialize DataTransformation with configuration and ingestion artifact.

        Parameters:
            data_transformation_config (DataTransformationConfig): Configuration for data transformation.
            data_ingestion_artifact (DataIngestionArtifact): Artifact containing paths to train and test data.
        """
        try:
            logging.info(f"{'>' * 20} Data Transformation {'<' * 20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise SrcException(e, sys)

    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Execute the data transformation process including outlier handling,
        correlation removal, dropping irrelevant features, and transforming
        categorical features using one-hot encoding.

        Returns:
            DataTransformationArtifact: Artifact containing paths to the transformed data and transformer object.
        """
        try:
            logging.info("Reading train and test data")
            # Load train and test data (assuming CSV files)
            main_df = pd.read_csv(self.data_ingestion_artifact.feature_store_file_path)
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            # Log the data shapes for verification
            logging.info(f"Main data shape: {main_df.shape}, Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")
            
            logging.info(f"Dropping Irrelevant Features from original data {features_to_drop}")
            main_df.drop(features_to_drop, axis=1, inplace=True)
            
            logging.info(f"Dropping Irrelevant Features from train and test data {features_to_drop}")
            # Drop irrelevant features
            train_df.drop(features_to_drop, axis=1, inplace=True)
            test_df.drop(features_to_drop, axis=1, inplace=True)
            
            logging.info("Handling outliers if available")
            # Handle outliers for main, train, and test data
            main_df[outliers_handling_features] = handling_outliers(main_df[outliers_handling_features])
            train_df[outliers_handling_features] = handling_outliers(train_df[outliers_handling_features])
            test_df[outliers_handling_features] = handling_outliers(test_df[outliers_handling_features])

            logging.info(f"Removing High Correlated Features from original data")
            # Remove highly correlated numerical features
            main_df = handle_num_correlations(main_df, threshold=self.data_transformation_config.correlation_threshold)
            
            logging.info(f"Removing High Correlated Features from training data")
            train_df = handle_num_correlations(train_df, threshold=self.data_transformation_config.correlation_threshold)
            
            logging.info(f"Removing High Correlated Features from testing data")
            test_df = handle_num_correlations(test_df, threshold=self.data_transformation_config.correlation_threshold)
            
            # Transform categorical features using One-Hot Encoding
            encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
            categorical_features = [feature for feature in train_df.columns if train_df[feature].dtype == "O"]
            logging.info(f"Applying One-Hot Encoding to these Categorical features {categorical_features}")

            # Fit encoder on main data and apply to train and test data
            main_encoded = encoder.fit_transform(main_df[categorical_features])
            main_encoded_df = pd.DataFrame(main_encoded, columns=encoder.get_feature_names_out())

            train_encoded = encoder.transform(train_df[categorical_features])
            train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out())

            test_encoded = encoder.transform(test_df[categorical_features])
            test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out())

            # Concatenate the encoded categorical features with the remaining numerical features
            logging.info("Concatenating encoded features with the rest of the data")
            main_input_features = pd.concat([main_df.drop(categorical_features, axis=1), main_encoded_df], axis=1)
            train_input_features = pd.concat([train_df.drop(categorical_features, axis=1), train_encoded_df], axis=1)
            test_input_features = pd.concat([test_df.drop(categorical_features, axis=1), test_encoded_df], axis=1)

            # Convert to numpy arrays
            main_arr = np.c_[main_input_features]
            train_arr = np.c_[train_input_features]
            test_arr = np.c_[test_input_features]

            # Log the shapes of the transformed arrays
            logging.info(f"Shape of transformed main data: {main_arr.shape}")
            logging.info(f"Shape of transformed train data: {train_arr.shape}")
            logging.info(f"Shape of transformed test data: {test_arr.shape}")

            logging.info(f"Saving transformed data arrays")
            # Save transformed data as numpy arrays
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_feature_store_data, array=main_arr
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path, array=train_arr
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path, array=test_arr
            )

            logging.info(f"Saving Transformer Object file")
            # Save the encoder object for future use
            save_object(
                self.data_transformation_config.transformer_object_file_path, obj=encoder
            )

            # Prepare and return the transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformer_object_file_path=self.data_transformation_config.transformer_object_file_path,
                transformed_feature_store_file_path=self.data_transformation_config.transformed_feature_store_data,
                transformed_train_data_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_data_file_path=self.data_transformation_config.transformed_test_file_path
            )
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise SrcException(e, sys)
