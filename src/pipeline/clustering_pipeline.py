from src.exception import SrcException
from src.logger import logging
from src.predictor import ModelResolver
import pandas as pd
import numpy as np
from src.utils import load_object, handle_num_correlations, handling_outliers
from src.config import outliers_handling_features, features_to_drop
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

FINAL_OUTPUT_DIR = "clustered_files"

def start_Cluster_prediction(input_file_path: str) -> str:
    """
    Executes batch prediction on the provided input CSV file.

    Steps:
      1. Reads the input file.
      2. Drops unnecessary features.
      3. Handles outliers and numerical correlations.
      4. Loads the trained transformer and model.
      5. Transforms the input data and predicts cluster labels.
      6. Saves the prediction results to a CSV file.

    Args:
        input_file_path (str): The file path of the input CSV file.

    Returns:
        str: The file path of the CSV file containing the predictions.
    """
    try:
        logging.info(f"{'>'*20} Starting Cluster Prediction {'<'*20}")
        os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

        # Check if file exists
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file '{input_file_path}' not found.")

        logging.info(f"Reading input file: {input_file_path}")
        df = pd.read_csv(input_file_path)

        if df.empty:
            raise ValueError("The input file is empty.")

        logging.info(f"Dropping unnecessary features: {features_to_drop}")
        df.drop(features_to_drop, axis=1, inplace=True)
        
        df_copy = df.copy()  #making copy of original data for preprocessing and training
        
        logging.info("Handling outliers and numerical correlations")
        if outliers_handling_features:
            df_copy[outliers_handling_features] = handling_outliers(df=df_copy[outliers_handling_features])
        
        df_copy = handle_num_correlations(df=df_copy)

        # Load transformer
        logging.info("Loading transformer for data transformation")
        transformer_path = ModelResolver(model_registry="saved_models").get_latest_transformer_path()
        logging.info(f"Transformer Path: {transformer_path}")
        transformer = load_object(file_path=transformer_path)

        # Feature transformation
        input_feature_names = list(transformer.feature_names_in_)
        transformed_array = transformer.transform(df_copy[input_feature_names])
        transformed_df = pd.DataFrame(
            transformed_array, columns=transformer.get_feature_names_out(input_feature_names)
        )

        input_encoded_df = pd.concat(
            [df_copy.drop(columns=input_feature_names), transformed_df], axis=1
        )

        # Load clustering model
        logging.info("Loading the best model for Clustering")
        model_path = ModelResolver(model_registry="saved_models").get_latest_model_path()
        logging.info(f"Model Path: {model_path}")
        model = load_object(file_path=model_path)

        # Predict clusters
        cluster_labels = model.predict(input_encoded_df)
        df["Cluster"] = cluster_labels

        # Assign names to clusters based on analysis in notebook\Preprocessing & Model_training.ipynb file
        # Used random seeding during model training to ensure consistent results matching the notebook analysis
        df["Cluster_Category"] = df["Cluster"].replace({0: "Low Spender", 1: "High Spender"}) 
        
        # Save results
        timestamp = datetime.now().strftime('%m%d%Y__%H%M%S')
        clustering_file_name = f"clustered_customer_{timestamp}.csv"
        clustering_file_path = os.path.join(FINAL_OUTPUT_DIR, clustering_file_name)

        df.to_csv(clustering_file_path, index=False, header=True) # Saving Predictions with Original Features and Labels
        logging.info(f"{'>'*20} Clustering Completed Successfully {'<'*20}")
        print(f'Clustered file saved as: {clustering_file_name}')
        
        return clustering_file_path

    except Exception as e:
        logging.error(f"Error in start_Cluster_prediction: {e}")
        raise SrcException(e, sys)
