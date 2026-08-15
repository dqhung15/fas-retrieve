FROM apache/airflow:2.8.3-python3.10

USER root
RUN apt-get update && \
  apt-get install -y git && \
  apt-get clean

USER airflow

RUN pip install --no-cache-dir \
  torch \
  torchvision \
  pillow \
  numpy \
  mlflow \
  psycopg2-binary \
  "email-validator>=2.0" \
  git+https://github.com/openai/CLIP.git

COPY --chown=airflow:root mlops/airflow/dags/ /opt/airflow/dags/
COPY --chown=airflow:root mlops/airflow/scripts/ /opt/airflow/scripts/
COPY --chown=airflow:root model_core/ /opt/airflow/model_core/