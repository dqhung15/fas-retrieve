import argparse
import json
import logging
import uuid
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main(args: argparse.Namespace):
  # 1. Connect to local Qdrant container
  client = QdrantClient("localhost", port=6333)
  collection_name = args.collection_name

  logger.info(f"Connected to Qdrant. Preparing collection: '{collection_name}'")

  # 2. Iterate over the embedding files
  input_dir = Path(args.input_dir)

  is_first_batch = True
  total_uploaded = 0

  for dress_type in args.dress_types:
    for split in args.splits:
      features_path = input_dir / f"{dress_type}_{split}_features.pt"
      names_path = input_dir / f"{dress_type}_{split}_names.json"

      if not features_path.exists() or not names_path.exists():
        logger.warning(f"Missing files for {dress_type} {split}, skipping...")
        continue

      features = torch.load(features_path)
      with open(names_path, "r", encoding="utf-8") as f:
        names = json.load(f)

      vector_dim = features.shape[1]

      # 3. Create the collection dynamically on the first file read
      if is_first_batch:
        if client.collection_exists(collection_name):
          logger.info("Collection exists. Dropping and recreating...")
          client.delete_collection(collection_name)

        client.create_collection(
          collection_name=collection_name,
          vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )

        is_first_batch = False

      # 4. Prepare data points
      points = []
      for i in range(len(names)):
        points.append(
          PointStruct(
            id = str(uuid.uuid4()),
            vector=features[i].tolist(),
            payload={
              "image_name": names[i],
              "dress_type": dress_type,
              "split": split,
            }
          )
        )

      # 5. Upload in batches
      client.upload_points(collection_name=collection_name, points=points)
      total_uploaded += len(points)
      logger.info(f"Uploaded {len(points)} points for {dress_type} ({split}).")
      
  logger.info(f"Success! Total points in Qdrant: {total_uploaded}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Upload embeddings to Qdrant.")
  parser.add_argument("--input-dir", type=str, default="data/embeddings")
  parser.add_argument("--collection-name", type=str, default="fashioniq")
  parser.add_argument("--dress-types", nargs="+", default=["dress", "shirt", "toptee"])
  parser.add_argument("--splits", nargs="+", default=["val", "test"])
  
  parsed_args = parser.parse_args()
  main(parsed_args)