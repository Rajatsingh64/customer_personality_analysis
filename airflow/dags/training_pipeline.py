import json
import pendulum
import os
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG(
    'Customer_Continuous_Training',
    default_args={'retries': 2},
    description='Customer Personality Analysis',
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2025, 3, 10, tz="UTC"),
    catchup=False,
    tags=['example'],
) as dag:

    def training(**kwargs):
        """Runs the training pipeline"""
        from src.pipeline.training_pipeline import run_training_pipeline
        run_training_pipeline()

    def sync_artifact_to_s3_bucket(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")
        os.system(f"aws s3 sync /app/artifact s3://{bucket_name}/artifacts")
        os.system(f"aws s3 sync /app/saved_models s3://{bucket_name}/saved_models")

    training_pipeline = PythonOperator(
        task_id="training_pipeline",
        python_callable=training
    )

    sync_data_to_s3 = PythonOperator(
        task_id="sync_data_to_s3",
        python_callable=sync_artifact_to_s3_bucket
    )

    training_pipeline >> sync_data_to_s3
