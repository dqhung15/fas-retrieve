# FAS-Retrieve: Composed Image Retrieval Pipeline

**FAS-Retrieve** is an end-to-end system designed for **Composed Image Retrieval (CIR)**. The project allows users to search for images using a combination of a reference image and a modifying text prompt (e.g., *a query image of a blue shirt + prompt "make it red"*).

This implementation leverages the architecture proposed in the paper **[CLIP4CIR: CLIP for Composed Image Retrieval](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/ABaldrati/CLIP4Cir&ved=2ahUKEwj0vMujwImWAxWNmlYBHcKQOPQQFnoECBkQAQ&usg=AOvVaw2NxBKAq6DgAJ-gUCYqy69L)**, combining CLIP-based visual and text embeddings via a specialized Combiner network to execute accurate vector search.

This repository serves as a hands-on implementation of MLOps practices, focusing heavily on model tracking, pipeline orchestration, and containerization. Through this project, I have learned:

* How to move from local environments.
* How to automate using Airflow.
* Gaining practical experience in setting up MLflow to systematically track experiments, making the fine-tuning process data-driven and organized.
* Understanding the networking and data flow between services.

---

## Working Process

### User inputs image and text

<img width="1611" height="767" alt="image" src="https://github.com/user-attachments/assets/3f4cf835-8d75-45ce-8a3c-a405e2db213c" />


## User clicks to an image

<img width="1587" height="248" alt="image" src="https://github.com/user-attachments/assets/27973443-80c1-444c-af70-32712b6a8926" />


## Fine-tune pipeline

<img width="1590" height="978" alt="image" src="https://github.com/user-attachments/assets/82b36bf3-2ee3-4f38-9c44-6135b620cc6a" />


---

## Getting Started

### Prerequisites

* Docker Engine
* Docker Compose 
* Python 3.10+ 

---

### Installation & Setup

#### 1. Clone the Repository

For image data and model checkpoint, I would use the FashionIQ dataset and the model from the paper as the original.

#### 2. Environment Configuration

Create a `.env` file in the root directory or set the required environment variables using .env.example as the template.

#### 3. Start Services

```bash
docker compose up -d
```

#### 4. Initialize Database Schema

```bash
docker exec -i ml_postgres psql -U admin -d tracking_db < scripts/schema.sql
```

#### 5. Extract Embeddings & Populate Vector Search

```bash
python -m model_core.extract_embeddings
python -m model_core.populate_qdrant
```

#### 6. Upload data to MinIO

```bash
python scripts/upload_data_to_minio.py
```

#### 7. Create Airflow Admin User

```bash
docker exec ml_airflow airflow users create \
  --username mlops \
  --firstname MLOps \
  --lastname Admin \
  --role Admin \
  --email mlops@example.com \
  --password admin
```
---

## Project Structure

```bash
fas-retrieve/
├── checkpoint/          # Model weights and CLIP4CIR checkpoints
├── data/                # Raw images and processing metadata
├── mlops/
│   ├── airflow/         # Airflow DAGs, scripts, and Docker configs
├── model_core/          # Embedding extraction, combiner logic, vector DB indexing
├── scripts/             # Setup scripts
├── services/
│   ├── api/             # FastAPI application
│   └── ui/              # Frontend web app static assets & Nginx config
├── docker-compose.yml   # Multi-container orchestration setup
```
