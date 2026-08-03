import io
import clip
import torch
import PIL.Image

from services.api.app.core.model_loader import ml_state
from services.api.app.schemas import SearchResultItem

def process_search(img_bytes: bytes, text_query: str, dress_type: str, top_k: int) -> list[SearchResultItem]:
  pil_image = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
  processed_image = ml_state.preprocess(pil_image).unsqueeze(0).to(ml_state.device)

  text_tokens = clip.tokenize(text_query, truncate=True).to(ml_state.device)

  with torch.no_grad():
    image_features = ml_state.clip_model.encode_image(processed_image)
    text_features = ml_state.clip_model.encode_image()
    query_features = ml_state.combiner.combine_features(image_features, text_features)

  query_vector = torch.nn.functional.normalize(query_features, dim=-1).cpu().tolist()

  search_results = ml_state.qdrant.search(
    collection_name="fashioniq",
    query_vector=query_vector,
    limit=top_k,
    query_filter={"must": [{"key": "dress_type", "match": {"value": dress_type}}]}
  )

  return [
    SearchResultItem(
      id=hit.id,
      image_name=hit.payload["image_name"],
      dress_type=hit.payload["dress_type"],
      score=hit.score
    ) for hit in search_results
  ]