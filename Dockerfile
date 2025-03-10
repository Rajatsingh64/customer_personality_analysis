# Use the latest Python 3.12 image as the base image
FROM python:3.12

# Set the user to root for installation purposes
USER root

# Create the app directory
RUN mkdir /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Set the working directory inside the container to /app
WORKDIR /app/

# Install dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Set environment variables for Airflow
ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

# Initialize the Airflow database
RUN airflow db init

# Create an Airflow admin user with the provided details
RUN airflow users create \
    -e rajat.k.singh64@gmail.com \
    -f Rajat \
    -l Singh \
    -p admin \
    -r Admin \
    -u admin

# Change permissions for the start.sh script to make it executable
RUN chmod 777 start.sh

# Install AWS CLI for interacting with AWS services
RUN apt-get update -y && apt-get install -y awscli

# Set the entrypoint to use shell
ENTRYPOINT [ "/bin/sh" ]

# Set the default command to run the start.sh script
CMD ["start.sh"]
