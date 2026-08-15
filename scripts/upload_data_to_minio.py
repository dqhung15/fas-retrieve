import os
from pathlib import Path
from dotenv import load_dotenv
from minio import Minio

load_dotenv(override=True)

DATASET_PATH = Path(os.getenv("DATA_DIR", "./data"), "fashionIQ_dataset", "images")
CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_DIR", "./checkpoint"))

if __name__ == "__main__":
  client = Minio(
    endpoint="localhost:9000",
    access_key=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
    secure=False,
  )

  if not client.bucket_exists('dataset'):
    client.make_bucket('dataset')
  if not client.bucket_exists('checkpoint'):
    client.make_bucket('checkpoint')
  if not client.bucket_exists('uploads'):
    client.make_bucket('uploads')

  for dirpath, dirnames, filenames in os.walk(DATASET_PATH):
    for filename in filenames:
      full_path = os.path.join(dirpath, filename)

      client.fput_object(
        'dataset',
        filename,
        full_path,
      )

  for dirpath, dirnames, filenames in os.walk(CHECKPOINT_PATH):
    for filename in filenames:
      full_path = os.path.join(dirpath, filename)

      client.fput_object(
        'checkpoint',
        filename,
        full_path,
      )

