import io
import logging
from typing import Annotated

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware

from services.api.app.model_loader import lifespan, ml_state
from services.api.app.schemas import SearchResponse, TrackInteractionRequest
from services.api.app.services.search_service import process_search
from services.api.app.services.tracking_service import log_interaction
from services.api.app.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
  title="Fashion Retrieval API",
  version="1.0.0",
  lifespan=lifespan
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)

@app.get("/")
async def root():
  return {"message": "Welcome to the Fashion Retrieval API. Visit /docs to test the endpoints."}

def get_storage_service(request: Request):
  return ml_state.obj_storage

@app.post("/search", response_model=SearchResponse)
async def search_fashion(
  image: Annotated[UploadFile, File(...)],
  text_query: Annotated[str, Form(...)],
  dress_type: Annotated[str, Form(...)],
  top_k: Annotated[int, Form()] = 10,
  storage = Depends(get_storage_service),
):
  try:
    img_bytes = await image.read()

    storage.put_object(
      bucket_name='uploads',
      object_name=image.filename,
      data=io.BytesIO(img_bytes),
      length=len(img_bytes),
      content_type=image.content_type, 
    )

    results = process_search(img_bytes, text_query, dress_type, top_k)
    
    return {"results": results, "message": "Success"}
      
  except Exception as e:
    logger.error("FATAL ERROR in /search route", exc_info=True)
    raise HTTPException(status_code=500, detail=str(e))

@app.post("/track")
async def track_interaction(payload: TrackInteractionRequest):
  try:
    log_interaction(payload)
    return {"status": "success", "message": "Interaction tracked."}
  except Exception as e:
    logger.error("FATAL ERROR in /track route", exc_info=True)
    raise HTTPException(status_code=500, detail="Database write failed.")

@app.post("/reload-model")
async def reload_model(storage = Depends(get_storage_service)):
  try:
    logger.info("Downloading latest checkpoint from MinIO...")
    storage.fget_object(
      bucket_name="checkpoint",
      object_name="combiner_latest.pt",
      file_path=settings.COMBINER_CHECKPOINT
    )
    
    logger.info("Loading new weights into memory...")
    checkpoint = torch.load(settings.COMBINER_CHECKPOINT, map_location=ml_state.device)
    ml_state.combiner.load_state_dict(checkpoint["Combiner"])
    
    ml_state.combiner.eval()
    
    logger.info("Model hot-swap complete.")
    return {"status": "success", "message": "Model weights dynamically reloaded."}
    
  except Exception as e:
    logger.error(f"FATAL ERROR during model reload: {str(e)}", exc_info=True)
    raise HTTPException(status_code=500, detail="Failed to reload model.")