from src.exception import SrcException
from src.logger import logging
from src.predictor import ModelResolver
import pandas as pd
import numpy as np
from src.utils import load_object, handle_num_correlations, handling_outliers
import os, sys
from datetime import datetime
from src.config import features_to_drop

PREDICTION_DIR = "Prediction"


def start_Cluster_prediction(input_file_path: str) -> str:
    """
    Executes batch prediction on the provided input CSV file.
    The function performs the following steps:
      1. Reads the input file.
      2. Drops unnecessary features.
      3. Handles outliers and numerical correlations.
      4. Loads the trained transformer and model.
      5. Transforms the input data and predicts cluster labels.
      6. Saves the prediction results to a CSV file in the prediction directory.

    Args:
        input_file_path (str): The file path of the input CSV file.

    Returns:
        str: The file path of the CSV file containing the predictions.
    """
    try:
        logging.info(f"{'>'*20} Starting Cluster Prediction {'<'*20}")
        os.makedirs(PREDICTION_DIR, exist_ok=True)

        logging.info("Creating ModelResolver object")
        model_resolver = ModelResolver(model_registry="saved_models")

        logging.info(f"Reading input file: {input_file_path}")
        df = pd.read_csv(input_file_path)

        # Remove unnecessary features based on configuration
        logging.info(f"Dropping unnecessary features: {features_to_drop}")
        df.drop(features_to_drop, axis=1, inplace=True)

        # Handle outliers and reduce high numerical correlations
        logging.info("Processing data: handling outliers and numerical correlations")
        df = handling_outliers(df=df)
        df = handle_num_correlations(df=df)

        # Load the transformer used during model training
        logging.info("Loading transformer for data transformation")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())

        # Retrieve input feature names used during training
        input_feature_names = list(transformer.feature_names_in_)
        logging.info(f"Transforming input features: {input_feature_names}")

        # Transform the data using the transformer and create a DataFrame of encoded features
        transformed_array = transformer.transform(df[input_feature_names])
        transformed_df = pd.DataFrame(
            transformed_array, 
            columns=transformer.get_feature_names_out(input_feature_names)
        )

        # Combine the transformed features with the remaining original features
        input_encoded_df = pd.concat(
            [df.drop(columns=input_feature_names).reset_index(drop=True), 
             transformed_df.reset_index(drop=True)],
            axis=1
        )

        # Load the best saved model for prediction
        logging.info("Loading the best model for prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())

        # Predict cluster labels using the loaded model
        cluster_labels = model.predict(input_encoded_df)
        df["Cluster"] = cluster_labels

        # Generate a unique filename for the predictions file using the current datetime
        prediction_file_name = os.path.basename(input_file_path).replace(
            ".csv", f"_{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv"
        )
        prediction_file_path = os.path.join(PREDICTION_DIR, prediction_file_name)

        # Save the predictions to a CSV file
        df.to_csv(prediction_file_path, index=False, header=True)
        logging.info(f"{'>'*20} Cluster Prediction Completed Successfully {'<'*20}")
        print(f'Prediction file >>>>>>>>>>>> {prediction_file_name}')
        
        return prediction_file_path

    except Exception as e:
        raise SrcException(e, sys)
