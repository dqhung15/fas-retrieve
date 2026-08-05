import io
import json
import logging
import urllib.request

import clip
import torch
import PIL.Image

from services.api.app.core.config import settings
from services.api.app.core.model_loader import ml_state
from services.api.app.schemas import SearchResultItem

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
  ch = logging.StreamHandler()
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  logger.addHandler(ch)

def process_search(img_bytes: bytes, text_query: str, dress_type: str, top_k: int) -> list[SearchResultItem]:
  logger.info(f"--- New Search Request Initiated ---")
  logger.info(f"Parameters: text_query='{text_query}', dress_type='{dress_type}', top_k={top_k}")
  
  try:
    logger.debug("Processing input image bytes...")
    pil_img = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
    processed_img = ml_state.preprocess(pil_img).unsqueeze(0).to(ml_state.device)
    
    logger.debug("Tokenizing text query...")
    text_tokens = clip.tokenize(text_query, truncate=True).to(ml_state.device)
    
    logger.info("Extracting and combining features via Combiner model...")
    with torch.no_grad():
      image_feature = ml_state.clip_model.encode_image(processed_img)
      text_feature = ml_state.clip_model.encode_text(text_tokens)
      query_feature = ml_state.combiner.combine_features(image_feature, text_feature).squeeze(0)
        
    query_vector = torch.nn.functional.normalize(query_feature, dim=-1).flatten().cpu().tolist()
    logger.debug(f"Generated query vector of length: {len(query_vector)}")
    
    # 4. Search Vector DB directly via REST
    url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}/collections/fashioniq/points/search"
    logger.info(f"Sending REST request to Qdrant at: {url}")
    
    payload = {
      "vector": query_vector,
      "limit": top_k,
      "filter": {
          "must": [{"key": "dress_type", "match": {"value": dress_type}}]
      },
      "with_payload": True
    }
    
    req = urllib.request.Request(
      url, 
      data=json.dumps(payload).encode('utf-8'), 
      headers={'Content-Type': 'application/json'}
    )
    
    with urllib.request.urlopen(req) as response:
      result_data = json.loads(response.read().decode('utf-8'))
        
    search_results = result_data.get("result", [])
    logger.info(f"Successfully retrieved {len(search_results)} results from Qdrant.")
    
    formatted_results = [
      SearchResultItem(
        id=str(hit["id"]),
        image_name=hit["payload"]["image_name"],
        dress_type=hit["payload"]["dress_type"],
        score=hit["score"]
      ) for hit in search_results
    ]
    
    logger.info("Search request completed successfully.")
    return formatted_results

  except Exception as e:
    logger.error(f"FATAL ERROR during process_search: {str(e)}", exc_info=True)
    raise e