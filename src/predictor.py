import os, sys
from glob import glob
from typing import Optional
from src.logger import logging
from src.exception import SrcException

# File names used in the project
file_name = "dataset/cleaned_customer.csv"
train_file_name = "train.csv"
test_file_name = "test.csv"
transformer_object_file_name = "transformer.pkl"
model_file_name = "model.pkl"


class ModelResolver:
    """
    The ModelResolver class is responsible for managing model versions stored in the model registry.
    It provides functionality to retrieve the latest model and transformer paths, as well as to
    generate new save paths for updated models.
    """

    def __init__(self, model_registry: str = "saved_models",
                 transformer_dir_name: str = "transformer",
                 model_dir_name: str = "model"):
        """
        Initializes the ModelResolver by setting up the model registry directory and directory names
        for the transformer and model.

        Args:
            model_registry (str): The root directory where model versions are stored.
            transformer_dir_name (str): The sub-directory name for saving transformer objects.
            model_dir_name (str): The sub-directory name for saving model objects.
        """
        try:
            self.model_registry = model_registry
            # Create the model registry directory if it does not exist
            os.makedirs(self.model_registry, exist_ok=True)
            self.transformer_dir_name = transformer_dir_name
            self.model_dir_name = model_dir_name
        except Exception as e:
            raise SrcException(e, sys)

    def get_latest_dir_path(self) -> Optional[str]:
        """
        Retrieves the directory path with the highest version number from the model registry.
        Assumes that each version directory is named with an integer.

        Returns:
            Optional[str]: The path to the latest version directory, or None if no directory exists.
        """
        try:
            # List all entries in the model registry directory
            dir_names = os.listdir(self.model_registry)
            if len(dir_names) == 0:
                return None

            # Convert directory names to integers to determine the latest version
            dir_names_int = list(map(int, dir_names))
            latest_dir_name = max(dir_names_int)
            return os.path.join(self.model_registry, f"{latest_dir_name}")
        except Exception as e:
            raise SrcException(e, sys)

    def get_latest_model_path(self):
        """
        Retrieves the file path for the latest saved model object.

        Returns:
            str: Full path to the latest model file.

        Raises:
            Exception: If no model directory is found.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception("Model is not available")
            return os.path.join(latest_dir, self.model_dir_name, model_file_name)
        except Exception as e:
            raise e

    def get_latest_transformer_path(self):
        """
        Retrieves the file path for the latest saved transformer object.

        Returns:
            str: Full path to the latest transformer file.

        Raises:
            Exception: If no transformer directory is found.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception("Transformer is not available")
            return os.path.join(latest_dir, self.transformer_dir_name, transformer_object_file_name)
        except Exception as e:
            raise e

    def get_latest_save_dir_path(self) -> str:
        """
        Generates a new directory path for saving a new model version. If no previous version
        exists, the version is set to 0.

        Returns:
            str: The new directory path to save the model.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                # If no directory exists, start with version "0"
                return os.path.join(self.model_registry, "0")
            latest_dir_num = int(os.path.basename(latest_dir))
            # Increment the latest version number for the new save directory
            return os.path.join(self.model_registry, f"{latest_dir_num + 1}")
        except Exception as e:
            raise e

    def get_latest_save_model_path(self):
        """
        Constructs the file path for saving the new model object.

        Returns:
            str: Full path where the new model file should be saved.
        """
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.model_dir_name, model_file_name)
        except Exception as e:
            raise e

    def get_latest_save_transformer_path(self):
        """
        Constructs the file path for saving the new transformer object.

        Returns:
            str: Full path where the new transformer file should be saved.
        """
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.transformer_dir_name, transformer_object_file_name)
        except Exception as e:
            raise e


class Predictor:
    """
    The Predictor class is designed to serve as a wrapper for making predictions using the
    resolved model and transformer objects. It leverages the ModelResolver to determine the
    correct model version to use for inference.
    """

    def __init__(self, model_resolver: ModelResolver):
        """
        Initializes the Predictor with a ModelResolver instance.

        Args:
            model_resolver (ModelResolver): An instance of ModelResolver to fetch model paths.
        """
        self.model_resolver = model_resolver
