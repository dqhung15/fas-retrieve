import os
import psycopg2
from contextlib import asynccontextmanager

import clip
import torch
from fastapi import FastAPI
from qdrant_client import QdrantClient
from minio import Minio

from model_core.combiner import Combiner
from model_core.data_utils import targetpad_transform
from services.api.app.config import settings

class AppState:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  clip_model = None
  preprocess = None
  combiner = None
  qdrant: QdrantClient = None
  db_conn = None
  obj_storage = None

ml_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
  # 1. Load CLIP
  ml_state.clip_model, _ = clip.load(settings.CLIP_MODEL_NAME, device=ml_state.device)
  ml_state.clip_model.eval()
  
  # 2. Setup Preprocess
  input_dim = ml_state.clip_model.visual.input_resolution
  ml_state.preprocess = targetpad_transform(1.25, input_dim)
  
  # 3. Load Combiner
  feature_dim = ml_state.clip_model.visual.output_dim
  ml_state.combiner = Combiner(clip_feature_dim=feature_dim, projection_dim=2560, hidden_dim=5120)

  try:
    print("Checking MinIO for updated combiner checkpoint...")
    ml_state.obj_storage.fget_object(
      bucket_name="checkpoint",
      object_name="combiner_latest.pt",
      file_path=settings.COMBINER_CHECKPOINT
    )
    print("Successfully downloaded latest checkpoint.")
  except Exception as e:
      print(f"No checkpoint found in MinIO or download failed: {e}")
  
  if os.path.exists(settings.COMBINER_CHECKPOINT):
    checkpoint = torch.load(settings.COMBINER_CHECKPOINT, map_location=ml_state.device)
    ml_state.combiner.load_state_dict(checkpoint["Combiner"])
    ml_state.combiner.to(ml_state.device)
    ml_state.combiner.eval()
  
  # 4. Connect to Databases
  ml_state.qdrant = QdrantClient(settings.QDRANT_HOST, port=settings.QDRANT_PORT)
  ml_state.db_conn = psycopg2.connect(
    dbname=settings.POSTGRES_DB,
    user=settings.POSTGRES_USER,
    password=settings.POSTGRES_PASSWORD,
    host=settings.POSTGRES_HOST,
    port=settings.POSTGRES_PORT
  )
  ml_state.obj_storage = Minio(
    endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin123"),
    secure=False,
  )
  
  yield
  
  ml_state.db_conn.close()
  del ml_state.clip_model
  del ml_state.combiner
  del ml_state.obj_storage
  torch.cuda.empty_cache()