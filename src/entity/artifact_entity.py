from dataclasses import dataclass
from typing import Optional


@dataclass
class DataIngestionArtifact:
    feature_store_file_path: Optional[str] = ""
    train_file_path: Optional[str] = ""
    test_file_path: Optional[str] = ""


@dataclass
class DataValidationArtifact:
    report_file_path: Optional[str] = ""


@dataclass
class DataTransformationArtifact:
    transformer_object_file_path: Optional[str] = ""
    transformed_train_data_file_path: Optional[str] = ""
    transformed_test_data_file_path: Optional[str] = ""


@dataclass
class ModelTrainingArtifact:
    model_object_file_path: Optional[str] = ""
    train_silhouette_score: Optional[float] = 0.0
    test_silhouette_score: Optional[float] = 0.0

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:Optional[float] = 0.0
    improved_accuracy:Optional[float] = 0.0   

@dataclass
class ModelPusherArtifact:
   pusher_model_dir:Optional[str] = ""
   saved_model_dir:Optional[str] = ""
   

