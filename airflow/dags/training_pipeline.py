import json
import os
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

# Define DAG for Customer Personality Analysis
with DAG(
    dag_id='customer_personality_analysis',  # Improved name for clarity
    default_args={'retries': 2},  # Retries in case of failure
    description='DAG for training and syncing Customer Personality Analysis data',
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2025, 3, 10, tz="UTC"),
    catchup=False,
    tags=['customer_analysis'],
) as dag:

    def training(**kwargs):
        """
        Executes the training pipeline for customer personality analysis.
        """
        from src.pipeline.training_pipeline import run_training_pipeline
        run_training_pipeline()

    def sync_artifact_to_s3_bucket(**kwargs):
        """
        Syncs training artifacts and models to an AWS S3 bucket.
        """
        bucket_name = os.getenv("BUCKET_NAME")  # Fetch bucket name from environment variables
        if bucket_name:
            os.system(f"aws s3 sync /app/artifact s3://{bucket_name}/artifacts")
            os.system(f"aws s3 sync /app/saved_models s3://{bucket_name}/saved_models")
        else:
            raise ValueError("BUCKET_NAME environment variable is not set.")

    # Task to run the training pipeline
    training_pipeline = PythonOperator(
        task_id="train_pipeline",
        python_callable=training,
        dag=dag,
    )

    # Task to sync data to S3
    sync_data_to_s3 = PythonOperator(
        task_id="sync_data_to_s3",
        python_callable=sync_artifact_to_s3_bucket,
        dag=dag,
    )

    # Define task dependencies
    training_pipeline >> sync_data_to_s3
