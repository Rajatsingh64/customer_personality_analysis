import json
from textwrap import dedent
import pendulum
import os
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG(
    'Customer_Cluster_Prediction',
    default_args={'retries': 2},
    description='Customer Personality Analysis',
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2025, 3, 10, tz="UTC"),
    catchup=False,
    tags=['example'],
) as dag:

    def download_files(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")
        input_dir = "/app/input_files"
        os.makedirs(input_dir, exist_ok=True)
        os.system(f"aws s3 sync s3://{bucket_name}/input_files {input_dir}")

    def cluster_prediction(**kwargs):
        from src.pipeline.clustering_pipeline import start_Cluster_prediction
        input_dir = "/app/input_files"
        for file_name in os.listdir(input_dir):
            start_Cluster_prediction(input_file_path=os.path.join(input_dir, file_name))

        
    def sync_prediction_dir_to_s3_bucket(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")
        #upload prediction folder to predictionfiles folder in s3 bucket
        os.system(f"aws s3 sync /app/clustered s3://{bucket_name}/clustered_files")
    
    download_input_files = PythonOperator(
        task_id="download_files",
        python_callable=download_files
    )

    generate_prediction_files = PythonOperator(
        task_id="generate_predictions",
        python_callable=cluster_prediction
    )

    upload_prediction_files = PythonOperator(
        task_id="upload_predictions",
        python_callable=sync_prediction_dir_to_s3_bucket
    )

    download_input_files >> generate_prediction_files >> upload_prediction_files
