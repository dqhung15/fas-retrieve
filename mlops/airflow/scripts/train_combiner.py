import os
import glob
import torch
import psycopg2
import mlflow
import clip
import PIL.Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from model_core.combiner import Combiner

DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_USER = os.getenv("POSTGRES_USER", "admin")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "admin123")
DB_NAME = os.getenv("POSTGRES_DB", "tracking_db")
UPLOAD_DIR = "/app/data/uploads"
DATASET_DIR = "/app/data/fashionIQ_dataset/images"
CHECKPOINT_PATH = "/app/checkpoint/combiner_latest.pt"
MIN_NUMBER_IMAGE = 5

class FeedbackDataset(Dataset):
  def __init__(self, interactions, preprocess):
    self.interactions = interactions
    self.preprocess = preprocess

  def __len__(self):
    return len(self.interactions)

  def _find_image(self, directory, base_name):
    search_pattern = os.path.join(directory, f"{base_name}.*")
    matches = glob.glob(search_pattern)
    
    if not matches:
      raise FileNotFoundError(f"Could not find any image matching {base_name} in {directory}")
      
    return matches[0]

  def __getitem__(self, index):
    row = self.interactions[index]
    ref_id, text_query, target_img_name = row

    ref_path = self._find_image(UPLOAD_DIR, ref_id)
    ref_img = self.preprocess(PIL.Image.open(ref_path).convert('RGB'))

    target_path = self._find_image(DATASET_DIR, target_img_name)
    target_img = self.preprocess(PIL.Image.open(target_path).convert('RGB'))

    text_tokens = clip.tokenize(text_query, truncate=True)

    return ref_img, text_tokens, target_img

def fetch_training_data():
  conn = psycopg2.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, dbname=DB_NAME)
  cursor = conn.cursor()

  query = """
    SELECT id, reference_image_name, query_text, selected_target_name
    FROM interactions
    WHERE selected_target_name IS NOT NULL
    AND is_used_for_training = FALSE
  """

  cursor.execute(query)
  raw_data = cursor.fetchall()

  if raw_data and len(raw_data) >= MIN_NUMBER_IMAGE:
    row_ids = tuple([row[0] for row in raw_data])
    
    update_query = """
      UPDATE interactions 
      SET is_used_for_training = TRUE 
      WHERE id IN %s
    """
    cursor.execute(update_query, (row_ids,))
    conn.commit()

  cursor.close()
  conn.close()

  data = [(row[1], row[2], row[3]) for row in raw_data]
  
  return data

def train():
  mlflow.set_tracking_uri("http://mlflow:5000")
  mlflow.set_experiment("Combiner-Fine-Tuning")

  interactions = fetch_training_data()
  if len(interactions) < MIN_NUMBER_IMAGE:
    print("Not enough data to train. Waiting for more user interactions.")
    return
  
  device = "cuda" if torch.cuda.is_available() else "cpu"
  clip_model, preprocess = clip.load("ViT-B/32", device=device)
  for param in clip_model.parameters():
    param.requires_grad = False

  dataset = FeedbackDataset(interactions=interactions, preprocess=preprocess)
  dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

  model = Combiner(clip_feature_dim=512, projection_dim=512, hidden_dim=1024).to(device)
  if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  criterion = torch.nn.TripletMarginLoss(margin=0.2)

  with mlflow.start_run():
    model.train()
    total_loss = 0.0

    for batch_idx, (ref_img, text_tokens, target_img) in enumerate(dataloader):
      ref_img, text_tokens, target_img = ref_img.to(device), text_tokens.to(device), target_img.to(device)

      optimizer.zero_grad()

      with torch.no_grad():
        ref_features = clip_model.encode_image(ref_img)
        text_features = clip_model.encode_text(text_tokens)
        target_features = clip_model.encode_image(target_img)

      combined_query_features = model.combine_features(ref_features, text_features)

      combined_query_features = F.normalize(combined_query_features, dim=-1)
      target_features = F.normalize(target_features, dim=-1)

      negative_features = torch.zeros_like(target_features)

      loss = criterion(combined_query_features, target_features, negative_features)
      loss.backward()
      optimizer.step()
      
      total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Training Complete. Average Loss: {avg_loss:.4f}")

    mlflow.log_metric("train_loss", avg_loss)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    mlflow.pytorch.log_model(model, "combiner_model")

if __name__ == "__main__":
  train()