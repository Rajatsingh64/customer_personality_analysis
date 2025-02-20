from src.predictor import ModelResolver
from src.entity.config_entity import ModelPusherConfig
from src.exception import SrcException
import os, sys
from src.utils import load_object, save_object
from src.logger import logging
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainingArtifact, ModelPusherArtifact


class ModelPusher:
    """
    The ModelPusher class is responsible for deploying the trained model and its associated
    transformer. It saves the objects both to a dedicated pusher directory and to a saved model
    directory, ensuring that the latest model is available for production use.
    """
    def __init__(self,
                 model_pusher_config: ModelPusherConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainingArtifact):
        """
        Initializes the ModelPusher with the configuration and transformation/training artifacts.

        Args:
            model_pusher_config (ModelPusherConfig): Configuration for model pushing.
            data_transformation_artifact (DataTransformationArtifact): Artifact produced from data transformation.
            model_trainer_artifact (ModelTrainingArtifact): Artifact produced from model training.
        """
        try:
            logging.info(f"{'>'*20} Initializing Model Pusher {'<'*20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            # Initialize the ModelResolver with the directory where the saved models are stored.
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise SrcException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Loads the trained model and its transformer, then saves them into both the pusher directory
        and the saved model directory. This process makes the model available for production deployment.

        Returns:
            ModelPusherArtifact: An artifact containing paths to the directories where the model
                                 and transformer are saved.
        """
        try:
            # Load the transformer and model objects using the file paths provided in the artifacts.
            logging.info("Loading transformer and trained model objects.")
            transformer = load_object(file_path=self.data_transformation_artifact.transformer_object_file_path)
            model = load_object(file_path=self.model_trainer_artifact.model_object_file_path)

            # Save the transformer and model to the model pusher directory.
            logging.info("Saving transformer and model to the model pusher directory.")
            save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)

            # Retrieve the latest save paths for the transformer and model from the ModelResolver.
            logging.info("Retrieving paths for saving objects in the saved model directory.")
            transformer_path = self.model_resolver.get_latest_save_transformer_path()
            model_path = self.model_resolver.get_latest_save_model_path()

            # Save the transformer and model to the saved model directory.
            logging.info("Saving transformer and model to the saved model directory.")
            save_object(file_path=transformer_path, obj=transformer)
            save_object(file_path=model_path, obj=model)

            # Create the ModelPusherArtifact containing the relevant directory paths.
            model_pusher_artifact = ModelPusherArtifact(
                pusher_model_dir=self.model_pusher_config.pusher_model_dir,
                saved_model_dir=self.model_pusher_config.saved_model_dir
            )
            logging.info(f"Model pusher artifact created: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise SrcException(e, sys)
