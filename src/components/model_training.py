import os
import sys
import re
import dill
import numpy as np
import pandas as pd
from typing import Optional
from src.entity import config_entity, artifact_entity
from src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import DataIngestionArtifact , ModelTrainingArtifact , DataTransformationArtifact
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from src.exception import SrcException
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.config import features_to_drop
from src.utils import save_object, load_object, save_numpy_array_data ,load_numpy_array_data
from src import utils
import warnings 
warnings.filterwarnings("ignore")


class ModelTrainer:
    """
    ModelTrainer is responsible for training and tuning clustering models.
    It selects the best model based on silhouette score for all clustering models.
    """

    def __init__(self, model_trainer_config: ModelTrainingConfig, 
                 data_ingestion_artifact: DataIngestionArtifact , 
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info(f'{">"*20} Model Trainer {"<"*20}')
            self.model_trainer_config = model_trainer_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise SrcException(e, sys)

    def tune_model(self, X, no_clusters: int = 10):
        """
        Tune clustering models to find the best number of clusters using GridSearchCV.
        
        Parameters:
        X (numpy.ndarray or pandas.DataFrame): Feature data for clustering.
        no_clusters (int): Maximum number of clusters to test (default is 10).
        
        Returns:
        best_model (KMeans): The best clustering model found.
        """
        try:
            logging.info("Tuning KMeans Model using GridSearchCV......")
            
            # Defining the parameter grid for n_clusters
            param_grid = {'n_clusters': list(range(2, no_clusters + 1))}  # Test values from 2 to no_clusters
            
            # Initialize the KMeans model
            model = KMeans(init="k-means++", n_init=10, random_state=42)

            # Setting up GridSearchCV with silhouette score as the scoring metric
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, 
                                       scoring=self.silhouette_scorer(X), n_jobs=-1)
            
            # Fit the model
            grid_search.fit(X)
            
            # Get the best model from the grid search
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_

            logging.info(f"Best Model found with K={best_model.n_clusters} and Silhouette Score={best_score}")
            
            return best_model  # Return the best-tuned model

        except Exception as e:
            raise SrcException(e,sys)
    
    def silhouette_scorer(self, X):
        """
        Custom silhouette scorer to be used in GridSearchCV.
        
        Parameters:
        X (numpy.ndarray or pandas.DataFrame): Feature data for clustering.
        
        Returns:
        silhouette_score: The silhouette score metric.
        """
        def scorer(estimator, X):
            labels = estimator.fit_predict(X)
            if len(set(labels)) > 1:  # Ensure more than one cluster
                return silhouette_score(X, labels)
            else:
                return float('-inf')  # Return negative infinity if only one cluster is found
        return scorer

    def initiate_model_training(self) -> ModelTrainingArtifact:

        try:
            
            logging.info(f"Loading original data array for fitting")
            main_array=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_feature_store_file_path)

            logging.info(f"Loading train and test array file")
            train_array = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_data_file_path)
            test_array = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_data_file_path)

            logging.info(f"Train the model")
            best_model=KMeans(n_clusters=2, init="k-means++" , n_init=10 , random_state=42 )
            best_model.fit(main_array)

            train_labels = best_model.predict(train_array)
            logging.info(f"Calculating train silhouette_score")
            train_silhouette_score = silhouette_score(train_array, train_labels)
            
            test_labels = best_model.predict(test_array)
            logging.info(f"Calculating test silhouette_score")
            test_silhouette_score = silhouette_score(test_array, test_labels)
            logging.info(f"train score: {train_silhouette_score} test score: {test_silhouette_score}")
            
            if test_silhouette_score < self.model_trainer_config.expected_silhouette_score:
                raise Exception(f"Model is not good as it is not able to give expected silhouette_score: {self.model_trainer_config.expected_silhouette_score}. Model actual score: {test_silhouette_score}")
            
            logging.info(f"Checking if our model is overfitting or not")    
            diff = abs(train_silhouette_score - test_silhouette_score)

            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and Test Model Difference: {diff} is more than overfitting Threshold: {self.model_trainer_config.overfitting_threshold}")
            
            # Save trained Model
            logging.info(f"Saving model Artifact")
            save_object(file_path=self.model_trainer_config.model_object_file_path, obj=best_model)

            # Prepare artifact
            logging.info(f"Prepare artifact")
            model_trainer_artifact = artifact_entity.ModelTrainingArtifact(
                model_object_file_path=self.model_trainer_config.model_object_file_path,
                train_silhouette_score=train_silhouette_score,
                test_silhouette_score=test_silhouette_score
            )
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            logging.error(f"Error in initiate_model_training: {e}")
            raise SrcException(e, sys)
