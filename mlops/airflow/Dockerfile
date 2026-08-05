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
  email-validator \
  git+https://github.com/openai/CLIP.git