import os

class Settings:
  # Postgres
  POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
  POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin123")
  POSTGRES_DB = os.getenv("POSTGRES_DB", "tracking_db")
  POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
  POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

  # Qdrant
  QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
  QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

  # Models
  CLIP_MODEL_NAME = "RN50x4"
  COMBINER_CHECKPOINT = os.getenv("COMBINER_CHECKPOINT", "fiq_comb_RN50x4_fullft.pt")

settings = Settings()