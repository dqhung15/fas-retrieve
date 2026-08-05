import argparse
import json
import logging
from pathlib import Path

import clip
import torch

from model_core.data_utils import FashionIQDataset, targetpad_transform, DEFAULT_DATA_DIR
from model_core.utils import extract_index_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initialized extraction on device: {device}")

    logger.info(f"Loading CLIP model ({args.clip_model})...")
    clip_model, _ = clip.load(args.clip_model, device=device)
    clip_model.eval()

    input_dim = clip_model.visual.input_resolution
    preprocess = targetpad_transform(1.25, input_dim)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dress_type in args.dress_types:
        for split in args.splits:
            logger.info(f"Processing FashionIQ | Category: {dress_type} | Split: {split}")

            dataset = FashionIQDataset(
                split=split,
                dress_types=[dress_type],
                mode="classic",
                preprocess=preprocess,
                data_dir=args.data_dir
            )

            features, names = extract_index_features(dataset, clip_model, device=device)

            features_path = output_dir / f"{dress_type}_{split}_features.pt"
            torch.save(features, features_path)
            
            names_path = output_dir / f"{dress_type}_{split}_names.json"
            with open(names_path, "w", encoding="utf-8") as f:
                json.dump(names, f)
            
            logger.info(f"Successfully saved {len(names)} features to {features_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save target image embeddings for Vector DB.")
    
    parser.add_argument(
        "--clip-model", 
        type=str, 
        default="RN50x4", 
        help="CLIP backbone version to load"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default=str(DEFAULT_DATA_DIR), 
        help="Root path to the dataset directory"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/embeddings", 
        help="Directory to save the extracted .pt and .json files"
    )
    parser.add_argument(
        "--dress-types", 
        type=str, 
        nargs="+", 
        default=["dress", "shirt", "toptee"], 
        help="List of FashionIQ categories to process"
    )
    parser.add_argument(
        "--splits", 
        type=str, 
        nargs="+", 
        default=["val", "test"], 
        help="Dataset splits to process"
    )
    
    parsed_args = parser.parse_args()
    main(parsed_args)