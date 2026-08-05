from pydantic import BaseModel
from typing import List, Optional

class SearchResultItem(BaseModel):
  id: str
  image_name: str
  dress_type: str
  score: float

class SearchResponse(BaseModel):
  results: List[SearchResultItem]
  message: str

class TrackInteractionRequest(BaseModel):
  session_id: str
  query_text: str
  reference_image_name: str
  selected_target_name: Optional[str] = None