FROM python:3.12

USER root
RUN mkdir /app
COPY . /app/
WORKDIR /app/

RUN pip3 install -r requirements.txt

ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

# Initialize Airflow database
RUN airflow db init

# Create Airflow user using GitHub Secrets
RUN airflow users create \
    -e "${AIRFLOW_EMAIL}" \
    -f Rajat \
    -l Singh \
    -p "${AIRFLOW_PASSWORD}" \
    -r Admin \
    -u "${AIRFLOW_USERNAME}"

RUN chmod 777 start.sh
RUN apt update -y && apt install awscli -y

ENTRYPOINT [ "/bin/sh" ]
CMD ["start.sh"]
