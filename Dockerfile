FROM python:3.8

USER root

# Install system dependencies
RUN apt-get update -y && apt-get install -y build-essential libssl-dev libffi-dev python3-dev gcc

# Upgrade pip
RUN pip install --upgrade pip

# Create app directory
RUN mkdir /app
COPY . /app/
WORKDIR /app/

# Install requirements
RUN pip install -r requirements.txt

# Set Airflow environment variables
ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

# Initialize Airflow database
RUN airflow db init

# Create an Airflow user
RUN airflow users create -e rajat.k.singh64@gmail.com -f Rajat -l Singh -p admin -r Admin -u admin

# Make start.sh executable
RUN chmod 777 start.sh

# Install AWS CLI
RUN apt-get update -y && apt-get install awscli -y

# Set the entrypoint and command
ENTRYPOINT ["/bin/sh"]
CMD ["start.sh"]
