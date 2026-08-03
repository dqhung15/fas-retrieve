from typing import Annotated
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from services.api.app.schemas import SearchResponse, TrackInteractionRequest
from services.api.app.services.search_service import process_search
from services.api.app.services.tracking_service import log_interaction

router = APIRouter(prefix="/api/v1")

@router.post("/search", response_model=SearchResponse)
async def search_fashion(
  image: Annotated[UploadFile, File(...)],
  text_query: Annotated[str, Form(...)],
  dress_type: Annotated[str, Form(...)],
  top_k: Annotated[int, Form(...)] = 10,
):
  try:
    img_bytes = await image.read()
    results = process_search(img_bytes, text_query, dress_type, top_k)
    return SearchResponse(results, message="Success")
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@router.post("/track")
async def track_interaction(payload: TrackInteractionRequest):
  try:
    log_interaction(payload)
    return {'status': 'success', 'message': 'Interaction tracked.'}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))