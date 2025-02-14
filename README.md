# 🧠 Customer Personality Analysis  
![Project Status](https://img.shields.io/badge/Project%20Status-Ongoing-orange)

## 📌 Overview  
Customer Personality Analysis helps businesses understand customer behavior, segment users, and provide personalized recommendations. This project utilizes **Machine Learning Operations (MLOps)** to streamline model training, deployment, and monitoring.

## 🚀 Technologies Used  
- **Machine Learning:** Scikit-Learn, XGBoost  
- **MLOps Tools:** Docker, Apache Airflow  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** AWS (EC2, S3, Lambda)  


## 🛠️ Features  
✔️ Customer segmentation using clustering techniques (K-Means, DBSCAN)  
✔️ Model pipeline automation with **Apache Airflow**  
✔️ Deployment on **AWS (EC2, S3, Lambda)**  
✔️ Scalable & secure cloud-based infrastructure  

## 📈 MLOps Workflow  
1️⃣ **Data Ingestion** - Extract customer data & preprocess it  
2️⃣ **Model Training** - Use ML models for segmentation  
3️⃣ **Deployment** - Deploy on **AWS EC2** using **Docker**  
4️⃣ **Automation** - Use **Apache Airflow** for pipeline scheduling  

## 🚀 AWS Deployment  
### 1️⃣ **Setup AWS EC2 Instance**  
- Launch an EC2 instance with Ubuntu  
- Install Docker & pull the project repository  

### 2️⃣ **Run the Project in Docker**  
```bash
# Clone the repository
git clone https://github.com/yourusername/Customer-Personality-Analysis.git

# Navigate to the project folder
cd Customer-Personality-Analysis

# Build and run Docker container
docker build -t customer-analysis .
docker run -p 8501:8501 customer-analysis
  
