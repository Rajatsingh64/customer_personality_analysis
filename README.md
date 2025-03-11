# Customer Personality Analysis  
![Project Status](https://img.shields.io/badge/Project%20Status-ongoing-orange)

## 📌 Overview  
Customer Personality Analysis helps businesses understand customer behavior, segment users, and provide personalized recommendations. This project leverages **Machine Learning Operations (MLOps)** to streamline model training, deployment, and monitoring.

## 📂 Project Navigation  
📁 [**Notebooks**](notebook/) | 📁 [**Pipelines**](src/pipeline/) | 📁 [**Airflow DAGs**](airflow/dags/) | 📁 [**Docs**](docs/)

## ✅ Completed Work  

### 🔄 EDA & Feature Engineering  
[![Completed EDA & FE](https://img.shields.io/badge/AlmostDone-EDA%20%26%20FE-blue)](notebook/EDA%20&%20Feature_Engineering.ipynb)  
[![Completed Preprocessing & Model Training](https://img.shields.io/badge/AlmostDone-Preprocessing%20%26%20Model%20Training-blue)](notebook/Preprocessing%20&%20Model_training.ipynb)

### 🔄 Machine Learning Pipelines  
[![Completed Training Pipeline](https://img.shields.io/badge/Completed-Training%20Pipeline-green)](src/pipeline/training_pipeline.py)  
[![Completed Clustering Prediction Pipeline](https://img.shields.io/badge/Completed-Cluster%20Pipeline-green)](src/pipeline/clustering_pipeline.py)

### 🔄 Airflow DAGs  
[![Completed Airflow DAG](https://img.shields.io/badge/Completed-Airflow%20DAG-green)](airflow/dags)  
- [`training_pipeline.py`](airflow/dags/training_pipeline.py)  
- [`clustering_pipeline.py`](airflow/dags/clustering_pipeline_dag.py)

## 📜 Architecture Documentation  
[![HLD (High-Level Design)](https://img.shields.io/badge/Available-HLD-blue)](docs/HLD.pdf)  
[![LLD (Low-Level Design)](https://img.shields.io/badge/Available-LLD-blue)](docs/LLD.pdf)  
[![DPR (Detailed Project Report)](https://img.shields.io/badge/Available-DPR-blue)](docs/DPR.pdf)

## 🔄 Ongoing Work  
[![Ongoing Work](https://img.shields.io/badge/In%20Progress-Ongoing-orange)](docs/Ongoing.md)  
- Model Performance Optimization  
- Deployment on Cloud  
- Monitoring & Logging  

```bash
# Docker activation in AWS CLI:
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
