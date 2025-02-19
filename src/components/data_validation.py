from src.utils import get_collection_as_dataframe
import pandas as pd
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from scipy.stats import ks_2samp, chi2_contingency
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.exception import SrcException
from typing import Optional
from src.utils import write_yaml_file, convert_columns_float
from src.config import date_column
import numpy as np
import os, sys

class DataValidation:

    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        """
        Initializes DataValidation class with configuration and artifacts.
        """
        try:
            logging.info(f"{'>'*20} Data Validation {'<'*20}")  # Log process start
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        
        except Exception as e:
            raise SrcException(e, sys)

    def drop_missing_values_columns(self, df: pd.DataFrame, report_key_name: str) -> Optional[pd.DataFrame]:
        """
        Drops columns with missing values exceeding the specified threshold.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame.
        report_key_name (str): Key name for reporting missing values.

        Returns:
        Optional[pd.DataFrame]: DataFrame after dropping columns or None if no columns remain.
        """
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isnull().sum() / df.shape[0]
            
            logging.info(f'Selecting columns with missing values above {threshold}') 
            drop_column_names = null_report[null_report > threshold].index

            logging.info(f"Columns to drop: {list(drop_column_names)}")
            self.validation_error[report_key_name] = list(drop_column_names)
            df.drop(list(drop_column_names), axis=1, inplace=True)
             
            # Return None if no columns remain
            if len(df.columns) == 0:
                return None
            
            return df

        except Exception as e:
            raise SrcException(e, sys)

    def is_required_columns_exist(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str) -> bool:
        """
        Checks if all required columns exist in the current DataFrame.
        
        Parameters:
        base_df (pd.DataFrame): Base dataset.
        current_df (pd.DataFrame): Current dataset.
        report_key_name (str): Key name for missing columns report.

        Returns:
        bool: True if all required columns exist, False otherwise.
        """
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = [col for col in base_columns if col not in current_columns]

            for col in missing_columns:
                logging.info(f"Column: [{col}] is missing.")
            
            if missing_columns:
                self.validation_error[report_key_name] = missing_columns
                return False
            return True
        
        except Exception as e:
            raise SrcException(e, sys)
    

    def data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str):
        try:
            drift_report = dict()

            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data = base_df[base_column]
                current_data = current_df[base_column]

                # Handle date features
                if base_data.dtype == 'object' and pd.to_datetime(base_data, format='%Y-%m-%d', errors='coerce').notnull().all():
                    # Convert to datetime
                    base_data = pd.to_datetime(base_data, errors='coerce')
                    current_data = pd.to_datetime(current_data, errors='coerce')

                    # Extract useful date features (e.g., year, month, day)
                    base_data_features = {
                        'year': base_data.dt.year,
                        'month': base_data.dt.month,
                        'day': base_data.dt.day,
                        'weekday': base_data.dt.weekday
                    }

                    current_data_features = {
                        'year': current_data.dt.year,
                        'month': current_data.dt.month,
                        'day': current_data.dt.day,
                        'weekday': current_data.dt.weekday
                    }

                    # Check for drift in extracted date features
                    for feature_name in base_data_features.keys():
                        base_feature = base_data_features[feature_name]
                        current_feature = current_data_features[feature_name]

                        # Perform KS Test to check for distribution drift
                        same_distribution = ks_2samp(base_feature, current_feature)
                        if same_distribution.pvalue > 0.05:
                            drift_report[f"{base_column}_{feature_name}"] = {
                                "pvalues": float(same_distribution.pvalue),
                                "same_distribution": True
                            }
                        else:
                            drift_report[f"{base_column}_{feature_name}"] = {
                                "pvalues": float(same_distribution.pvalue),
                                "same_distribution": False
                            }

                # Handle numerical features
                elif base_data.dtype in [np.float64, np.int64]:
                    logging.info(f"Hypothesis {base_column}: {base_data.dtype}, {current_data.dtype}")
                    same_distribution = ks_2samp(base_data, current_data)

                    if same_distribution.pvalue > 0.05:
                        drift_report[base_column] = {
                            "pvalues": float(same_distribution.pvalue),
                            "same_distribution": True
                        }
                    else:
                        drift_report[base_column] = {
                            "pvalues": float(same_distribution.pvalue),
                            "same_distribution": False
                        }

                # Handle categorical features
                elif base_data.dtype == 'object':
                    # Creating contingency table
                    contingency_table = pd.crosstab(base_data, current_data)
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)	

                    if p_value > 0.05:
                        drift_report[base_column] = {
                            "pvalues": float(p_value),
                            "same_distribution": True
                        }
                    else:
                        drift_report[base_column] = {
                            "pvalues": float(p_value),
                            "same_distribution": False
                        }

            self.validation_error[report_key_name] = drift_report
        except Exception as e:
            raise Exception(f"Error during drift detection: {str(e)}")


    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiates the data validation process.

        Returns:
        DataValidationArtifact: Object containing the validation report file path.
        """
        try:
            logging.info("Reading base dataset")
            base_df = pd.read_csv(self.data_validation_config.base_dataset_file_path)
            base_df.replace({"na": np.NAN}, inplace=True)

            logging.info("Dropping missing value columns from base dataset")
            base_df = self.drop_missing_values_columns(base_df, "missing_values_within_base_dataset")

            logging.info("Reading train dataset")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)

            logging.info("Reading test dataset")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info("Dropping missing value columns from train dataset")
            train_df = self.drop_missing_values_columns(train_df, "missing_values_within_train_dataset")

            logging.info("Dropping missing value columns from test dataset")
            test_df = self.drop_missing_values_columns(test_df, "missing_values_within_test_dataset")

            # Convert columns to float for numerical analysis
            logging.info("Converting columns to float format")
            base_df = convert_columns_float(base_df)
            train_df = convert_columns_float(train_df)
            test_df = convert_columns_float(test_df)

            # Check if all required columns exist in train and test datasets
            logging.info("Checking if all required columns exist in train dataset")
            train_df_columns_status = self.is_required_columns_exist(base_df, train_df, "missing_columns_within_train_dataset")

            logging.info("Checking if all required columns exist in test dataset")
            test_df_columns_status = self.is_required_columns_exist(base_df, test_df, "missing_columns_within_test_dataset")

            # Detect data drift if columns exist
            if train_df_columns_status:
                logging.info("Detecting data drift in train dataset")
                self.data_drift(base_df, train_df, "data_drift_within_train_dataset")

            if test_df_columns_status:
                logging.info("Detecting data drift in test dataset")
                self.data_drift(base_df, test_df, "data_drift_within_test_dataset")

            # Write validation report to YAML file
            logging.info("Writing validation report to YAML file")
            write_yaml_file(file_path=self.data_validation_config.report_file_path, data=self.validation_error)

            # Return validation artifact
            data_validation_artifact = DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            logging.info(f"Data validation artifact created: {data_validation_artifact}")

            return data_validation_artifact
        
        except Exception as e:
            raise SrcException(e, sys)
