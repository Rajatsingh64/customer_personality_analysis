# Customer Personality Analysis  
![Project Status](https://img.shields.io/badge/Project%20Status-ongoing-orange)

## 📌 Overview  
Customer Personality Analysis helps businesses understand customer behavior, segment users, and provide personalized recommendations. This project leverages **Machine Learning Operations (MLOps)** to streamline model training, deployment, and monitoring.

## ⚠ Important Note  
- **Python Version:** Please ensure you are using **Python 3.12** to maintain compatibility with the project's dependencies and requirements.

---

## 📂 Project Navigation  
📁 [**Notebooks**](notebooks/) | 📁 [**Pipelines**](src/pipeline/) | 📁 [**Airflow DAGs**](airflow/dags/) | 📁 [**Docs**](docs/) | 📁 [**Deployment**](deployment/)

---
<h2 align="">Tools and Technologies Used</h2>
<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="Scikit-learn" height="60">&nbsp;&nbsp;&nbsp;
    <img src="https://upload.wikimedia.org/wikipedia/commons/8/84/Matplotlib_icon.svg" alt="Matplotlib" height="60">&nbsp;&nbsp;&nbsp;
    <img src="https://a0.awsstatic.com/libra-css/images/logos/aws_logo_smile_1200x630.png" alt="AWS" height="60">&nbsp;&nbsp;&nbsp;
    <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Docker_%28container_engine%29_logo.svg" alt="Docker" height="60">&nbsp;&nbsp;&nbsp;
    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" height="60">&nbsp;&nbsp;&nbsp;
    <img src="https://seaborn.pydata.org/_images/logo-wide-lightbg.svg" alt="Seaborn" height="60">&nbsp;&nbsp;&nbsp;
    <img src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg" alt="Pandas" height="60">&nbsp;&nbsp;&nbsp;
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg" alt="NumPy" height="60">&nbsp;&nbsp;&nbsp;
    <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit" height="60">&nbsp;&nbsp;&nbsp;
    <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python" height="60">&nbsp;&nbsp;&nbsp;
    <img src="https://www.mongodb.com/assets/images/global/favicon.ico" alt="MongoDB" height="60">&nbsp;&nbsp;&nbsp;
    <img src="https://airflow.apache.org/docs/apache-airflow/stable/_images/apache-airflow-icon.png" alt="Apache Airflow" height="60">
</p>

Customer-Personality-Analysis/
## 💁️ Project Structure
```
Restaurant-Rating-Prediction/
│
├── .dockerignore                     # 🚫 Ignore files for Docker
├── .env                              # 🔑 Environment variables
├── .gitignore                        # 🚫 Ignore files for Git
│
├── .github/
│   └── workflows/
│       └── main.yaml                 # ⚙️ GitHub Actions CI/CD pipeline
│
├── airflow/                          # 💨 Apache Airflow DAGs
│   └── dags/                         # 📅 Workflow DAGs
│       ├── clustering_pipeline.py# 🔍 Airflow DAG for clustering
│       └── training_pipeline.py      # 🎯 Airflow DAG for model training
│
├── artifact/                        # 🐂 Contains all intermediate and final outputs
├── clustered_files/                  # 📂 Clustered processed files
├── data_dump.py                      # 🛋️ Dumps data into MongoDB Atlas
├── docker-compose.yml                # 🔧 Docker Compose for multi-container setup
├── Dockerfile                        # 💪 Docker image setup
│
├── LICENSE                           # 📚 MIT License file
├── main.py                           # 🚀 Entry point for training and predictions
├── notebook/                         # 📚 Jupyter notebooks
│   ├── EDA & Feature_Engineering.ipynb        # 🔄 Exploratory Data Analysis
│   ├── Preprocessing & Model_training.ipynb   # 🎓 Model training steps
│
├── README.md                         # 📖 Project documentation
├── requirements.txt                  # 📌 Dependencies for the project
├── saved_models/                     # 🎯 Production-ready models and transformers
├── setup.py                          # ⚙️ Package setup for `src`
│
├── src/
│   ├── components/                   # 🏢 Core pipeline components
│   │   ├── data_ingestion.py         # 📅 Handles data collection
│   │   ├── data_transformation.py    # 🔄 Prepares data for training
│   │   ├── data_validation.py        # ✅ Validates raw data
│   │   ├── model_evaluation.py       # 📊 Evaluates the model
│   │   ├── model_pusher.py           # 🚀 Pushes the trained model to deployment
│   │   ├── model_training.py         # 🎓 Trains the machine learning model
│   │
│   ├── config.py                     # ⚙️ Configuration management and environment variables
│   ├── entity/                       # 📆 Data structures for pipeline
│   │   ├── artifact_entity.py        # 🐂 Artifacts generated by pipeline stages
│   │   └── config_entity.py          # ⚙️ Configuration-related entities
│   │
│   ├── exceptions.py                 # ❗ Custom exception handling
│   ├── logger.py                     # 💜 Logging setup
│   ├── pipeline/                     # 🔄 Pipeline automation
│   │   ├── clustering_pipeline.py    # 🔍 Handles clustering predictions
│   │   └── training_pipeline.py      # 🎯 Automates training workflow
│   │
│   └── utils.py                      # 🛠️ Utility functions
```

### 🔄 EDA & Feature Engineering  
[![Completed EDA & FE](https://img.shields.io/badge/Completed-EDA%20%26%20FE-green)](notebooks/EDA%20&%20Feature_Engineering.ipynb)  
[![Completed Preprocessing & Model Training](https://img.shields.io/badge/Completed-Preprocessing%20%26%20Model%20Training-green)](notebooks/Preprocessing%20%26%20Model_training.ipynb)

### 🔄 Machine Learning Pipelines  
[![Completed Training Pipeline](https://img.shields.io/badge/Completed-Training%20Pipeline-green)](src/pipeline/training_pipeline.py)  
[![Completed Clustering Prediction Pipeline](https://img.shields.io/badge/Completed-Cluster%20Pipeline-green)](src/pipeline/cluster_prediction_pipeline.py)

### 🔄 Airflow DAGs  
[![Completed Airflow DAG](https://img.shields.io/badge/Completed-Airflow%20DAG-green)](airflow/dags)  
- [`training_pipeline.py`](airflow/dags/training_pipeline.py)  
- [`clustering_pipeline_dag.py`](airflow/dags/clustering_pipeline_dag.py)

---

## 📜 Architecture Documentation  
[![HLD (High-Level Design)](https://img.shields.io/badge/Ongoing-HLD-blue)](docs/HLD.pdf)  
[![LLD (Low-Level Design)](https://img.shields.io/badge/Ongoin-LLD-blue)](docs/LLD.pdf)  
[![DPR (Detailed Project Report)](https://img.shields.io/badge/Ongoin-DPR-blue)](docs/DPR.pdf)

---

## 🚀 Step-by-Step Deployment Guide  

### 1️⃣ **AWS EC2 Instance Setup**  
📌 **Steps:**  
- Create an EC2 instance  
- Configure security groups  
- SSH into the instance  

  
**Install Docker:**
   SSH into your EC2 instance and run the following commands:

   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   newgrp docker
   docker --version
  ```

📺 **GIF Demo:**  
![EC2 Runner Activation](deployment/gifs/ec2_setup.gif)

> **GitHub Runner Setup(Example):**  
> After SSH-ing into your EC2 instance, set up the GitHub self-hosted runner by executing the following commands:

> ```bash
> # Create a directory for the GitHub Actions runner and navigate into it
> mkdir actions-runner && cd actions-runner
> 
> # Download the latest runner package (update the version as needed)
> curl -o actions-runner-linux-x64.tar.gz -L https://github.com/actions/runner/releases/download/v2.300.0/actions-runner-linux-x64-2.300.0.tar.gz
> 
> # Extract the runner package
> tar xzf actions-runner-linux-x64.tar.gz
> 
> # Configure the runner (replace <REPO_URL> and <RUNNER_TOKEN> with your repository URL and runner token)
> ./config.sh --url https://github.com/<your-username>/<your-repository> --token <RUNNER_TOKEN>
> 
> # Start the runner
> ./run.sh
> ```
>  
> *Note: Ensure you have generated a runner token from your GitHub repository's settings under "Actions" → "Runners" → create "New-self-hosted-runner" → "linux".*

### 3️⃣ **IAM Role & Access Key Setup**  
📌 **Steps:**  
- Create an IAM user  
- Attach necessary policies (e.g., `S3FullAccess`, `EC2FullAccess` . `AdministratorAcess`)  
- Generate `AWS_ACCESS_KEY_ID` & `AWS_SECRET_ACCESS_KEY` 

 ### 2️⃣ **AWS S3 Bucket Creation**  
📌 **Steps:**  
- Navigate to AWS S3  
- Create a bucket  
- Set access permissions  

## 🐳 AWS ECR Creation Steps
### 1️⃣ **Create ECR Repository**  
- Open the **Amazon ECR console** at https://console.aws.amazon.com/ecr.
- Select **Repositories** from the left menu and click on **Create repository**.
- Choose a **private repository** type and provide a **repository name** (e.g., `customer-personality-analysis`).
- Click **Create repository**.

### 4️⃣ **GitHub Repository & Secrets Configuration**  
📌 **Steps:**  
- Create a GitHub repository  
- Navigate to `Settings` → `Secrets and variables` → `Actions`  
- Add the following secrets:  

  | Secret Name              | Description                      |
  |--------------------------|----------------------------------|
  | `AWS_ACCESS_KEY_ID`      | AWS Access Key                   |
  | `AWS_SECRET_ACCESS_KEY`  | AWS Secret Access Key            |
  | `AWS_REGION`             | AWS Region (e.g., `us-east-1`)     |
  | `BUCKET_NAME`            | S3 Bucket Name                   |
  | `MONGO_DB_URL`           | MongoDB Connection URL           |
  | `ECR_REPOSITORY_NAME`    | AWS ECR Repository Name          |
  | `AWS_ECR_LOGIN_URI`      | AWS ECR Login URI                |

📺 **GIF Demo:**  
![GitHub Secrets](deployment/gifs/github_secrets.gif)  

### 5️⃣ **GitHub Actions CI/CD (`main.yml`)**  
📌 **Steps:**  
- Configure `.github/workflows/main.yml`  
- Automate deployment to AWS  

📄 **GitHub Actions Workflow:**  
[![View GitHub Actions Workflow](https://img.shields.io/badge/View-Main.yml-blue?logo=github)](.github/workflows/main.yml)


