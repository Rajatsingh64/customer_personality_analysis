FROM python:3.12

# Use build arguments for sensitive data during build time
ARG AIRFLOW_EMAIL
ARG AIRFLOW_USERNAME
ARG AIRFLOW_PASSWORD

USER root
RUN mkdir /app
COPY . /app/
WORKDIR /app/

# Install dependencies
RUN pip3 install -r requirements.txt

# Set Airflow configuration environment variables
ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

# Initialize Airflow DB
RUN airflow db init

# Create Airflow user using build-time arguments
RUN airflow users create \
    --email "${AIRFLOW_EMAIL}" \
    --first "Rajat" \
    --last "Singh" \
    --password "${AIRFLOW_PASSWORD}" \
    --role "Admin" \
    --username "${AIRFLOW_USERNAME}" \
    --no-input  # Ensures no interactive input is requested

# Allow script execution
RUN chmod 777 start.sh

# Install AWS CLI
RUN apt update -y && apt install awscli -y

# Set entrypoint to start Airflow
ENTRYPOINT [ "/bin/sh" ]
CMD ["start.sh"]
