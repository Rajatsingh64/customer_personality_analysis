import os
import pendulum
import subprocess
from airflow import DAG
from airflow.operators.python import PythonOperator

# Define DAG for Customer Personality Analysis
with DAG(
    dag_id='customer_personality_analysis',  # Improved naming
    default_args={'retries': 2},  # Retries in case of failure
    description='DAG for batch predictions in Customer Personality Analysis',
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2025, 3, 10, tz="UTC"),
    catchup=False,
    tags=['customer_analysis'],
) as dag:

    def download_files(**kwargs):
        """
        Downloads input files from an S3 bucket to the local directory.
        """
        bucket_name = os.getenv("BUCKET_NAME")
        input_dir = "/app/input_files"

        if not bucket_name:
            raise ValueError("BUCKET_NAME environment variable is not set.")

        # Create input directory if not exists
        os.makedirs(input_dir, exist_ok=True)

        # Sync input files from S3
        subprocess.run(
            ["aws", "s3", "sync", f"s3://{bucket_name}/input_files", input_dir],
            check=True
        )

    def batch_prediction(**kwargs):
        """
        Runs batch predictions on all input files using the clustering pipeline.
        """
        from src.pipeline.clustering_pipeline import start_Cluster_prediction
        input_dir = "/app/input_files"

        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

        for file_name in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file_name)
            if os.path.isfile(file_path):
                start_Cluster_prediction(input_file_path=file_path)

    def sync_prediction_dir_to_s3_bucket(**kwargs):
        """
        Uploads the generated prediction files to the S3 bucket.
        """
        bucket_name = os.getenv("BUCKET_NAME")
        prediction_dir = "/app/prediction"

        if not bucket_name:
            raise ValueError("BUCKET_NAME environment variable is not set.")

        if not os.path.exists(prediction_dir):
            raise FileNotFoundError(f"Prediction directory {prediction_dir} does not exist.")

        subprocess.run(
            ["aws", "s3", "sync", prediction_dir, f"s3://{bucket_name}/prediction_files"],
            check=True
        )

    # Task to download input files from S3
    download_input_files = PythonOperator(
        task_id="download_files",
        python_callable=download_files,
    )

    # Task to generate batch predictions
    generate_prediction_files = PythonOperator(
        task_id="generate_predictions",
        python_callable=batch_prediction,
    )

    # Task to upload prediction results to S3
    upload_prediction_files = PythonOperator(
        task_id="upload_predictions",
        python_callable=sync_prediction_dir_to_s3_bucket,
    )

    # Define task dependencies
    download_input_files >> generate_prediction_files >> upload_prediction_files
