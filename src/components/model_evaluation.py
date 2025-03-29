from src.entity import artifact_entity, config_entity
from src.exception import SrcException
from src.predictor import ModelResolver
from src.logger import logging
from src.entity import artifact_entity
from src.config import features_to_drop , outliers_handling_features
import os, sys
from src import utils
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import warnings 
warnings.filterwarnings("ignore")


class ModelEvaluation:
    def __init__(self,
                 model_evaluation_config: config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainingArtifact):
        try:
            logging.info(f'{">"*20} Model Evaluation {"<"*20}')
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            # Initialize the ModelResolver to fetch the latest production model if it exists
            self.model_resovler = ModelResolver()
        except Exception as e:
            raise SrcException(e, sys)

    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        """
        Evaluates the newly trained model by comparing it against the production model.
        If a previously saved production model is found, both models are evaluated on the test
        dataset using the silhouette score. The new model is accepted only if its performance
        improves over the production model.
        
        Returns:
            ModelEvaluationArtifact: An artifact containing the evaluation results including
            whether the new model is accepted and the improvement in accuracy.
        """
        try:
            # Check if a previously saved production model exists for comparison.
            logging.info("Checking if a saved production model exists for performance comparison.")
            latest_dir_path = self.model_resovler.get_latest_dir_path()
            if latest_dir_path==None:
                # No production model found; accept the new model by default.
                model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(
                    is_model_accepted=True,
                    improved_accuracy=None
                )
                logging.info(f"Model Evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

            # Retrieve the paths for the production transformer and model.
            logging.info("Locating the production transformer and model.")
            transformer_path = self.model_resovler.get_latest_transformer_path()
            model_path = self.model_resovler.get_latest_model_path()

            # Load the production transformer and model objects.
            logging.info("Loading production transformer and model objects.")
            transformer = utils.load_object(file_path=transformer_path)
            model = utils.load_object(file_path=model_path)

            # Load the current (newly trained) model and transformer objects.
            logging.info("Loading newly trained model and transformer objects.")
            current_model = utils.load_object(file_path=self.model_trainer_artifact.model_object_file_path)
            current_transformer = utils.load_object(file_path=self.data_transformation_artifact.transformer_object_file_path)

            # Load the test dataset for model evaluation, ensuring consistency with the data transformation process.
            logging.info("Loading test dataset from %s", self.data_ingestion_artifact.test_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Remove unnecessary features from the dataset (e.g., columns with high correlation or outlier influence).
            logging.info("Dropping unnecessary features: %s", features_to_drop)
            test_df.drop(features_to_drop, axis=1, inplace=True)

            # Handle outliers to stabilize the dataset and improve model performance.
            logging.info("Handling outliers in the test dataset during model evaluation")
            test_df[outliers_handling_features] = utils.handling_outliers(test_df[outliers_handling_features])

            # Address high numerical correlations to reduce multicollinearity issues.
            test_df = utils.handle_num_correlations(test_df)

            # Process the test data using the production transformer to ensure consistent feature engineering.
            # Retrieve the feature names that were used during production training (e.g., after applying OneHotEncoder).
            input_features_name = list(transformer.feature_names_in_)
            # Transform the test data's categorical features.
            test_encoded = transformer.transform(test_df[input_features_name])
            # Create a DataFrame for the encoded features with the appropriate column names.
            test_df_encoded = pd.DataFrame(test_encoded, columns=transformer.get_feature_names_out(input_features_name))
            # Combine the remaining original features with the newly encoded features.
            input_df = pd.concat(
                [test_df.drop(columns=input_features_name).reset_index(drop=True),
                 test_df_encoded.reset_index(drop=True)],
                axis=1
            )

            # Evaluate the production model using the test data.
            y_hat = model.predict(input_df)
            previous_model_score = silhouette_score(input_df, y_hat)
            logging.info(f"Silhouette score for production model: {previous_model_score}")

            # Evaluate the newly trained model using the same test data.
            y_hat = current_model.predict(input_df)
            current_model_score = silhouette_score(input_df, y_hat)
            logging.info(f"Silhouette score for newly trained model: {current_model_score}")

            # If the new model does not outperform the production model, reject it.
            if current_model_score <previous_model_score:
                logging.info("The newly trained model does not outperform the current production model.")
                raise Exception("Current trained model is not better than the previous model")

            # Calculate the improvement in performance and create the evaluation artifact.
            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                is_model_accepted=True,
                improved_accuracy=current_model_score - previous_model_score
            )
            logging.info(f"Model Evaluation artifact: {model_eval_artifact}")
            return model_eval_artifact

        except Exception as e:
            raise SrcException(e, sys)
