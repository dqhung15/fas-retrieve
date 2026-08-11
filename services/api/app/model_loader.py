import os
import psycopg2
from contextlib import asynccontextmanager

import clip
import torch
from fastapi import FastAPI
from qdrant_client import QdrantClient

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
  
  yield
  
  # Teardown
  ml_state.db_conn.close()
  del ml_state.clip_model
  del ml_state.combiner
  torch.cuda.empty_cache()