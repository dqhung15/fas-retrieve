# FAS-Retrieve: Composed Image Retrieval Pipeline

**FAS-Retrieve** is an end-to-end system designed for **Composed Image Retrieval (CIR)**. The project allows users to search for images using a combination of a reference image and a modifying text prompt (e.g., *a query image of a blue shirt + prompt "make it red"*).

This implementation leverages the architecture proposed in the paper **[CLIP4CIR: CLIP for Composed Image Retrieval](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/ABaldrati/CLIP4Cir&ved=2ahUKEwj0vMujwImWAxWNmlYBHcKQOPQQFnoECBkQAQ&usg=AOvVaw2NxBKAq6DgAJ-gUCYqy69L)**, combining CLIP-based visual and text embeddings via a specialized Combiner network to execute accurate vector search.

---

## Architecture & Features

* **Vector Search Database:** **Qdrant** for storing high-dimensional embeddings and serving fast similarity queries.
* **Relational Database:** **PostgreSQL** for tracking data, metadata, and backend state.
* **Workflow Orchestration:** **Apache Airflow** for managing ETL pipelines, embedding extraction, and Qdrant indexing.
* **Model Tracking:** **MLflow** for experiment tracking, parameter monitoring, and artifact storage.
* **Serving & Frontend:** **FastAPI** backend coupled with an **Nginx-served UI** for real-time retrieval.

---

## What I Learned

Building this MLOps pipeline provided practical experience in scaling research models into production-ready software:

1. **Multimodal Embeddings & CIR:** Implementing CLIP4CIR architectures to project relative image-text edits into a unified latent space.
2. **Vector Indexing & Retrieval:** Setting up and populating Qdrant collections for scalable, real-time nearest-neighbor search.
3. **Pipeline Orchestration:** Automating offline data pipelines (embedding extraction, index generation) using Airflow DAGs.
4. **Experiment Management:** Tracking model artifacts, checkpoints, and metrics centrally using MLflow.
5. **Containerized MLOps Infrastructure:** Orchestrating multiple microservices (API, UI, MLflow, Airflow, Postgres, Qdrant) into a single cohesive network using Docker Compose.

---

## Getting Started

### Prerequisites

* Docker Engine (v20.10 or higher)
* Docker Compose (v2.0 or higher)
* Python 3.10+ (optional, for local development)

---

### Installation & Setup

#### 1. Clone the Repository

#### 2. Environment Configuration

Create a `.env` file in the root directory or set the required environment variables:

```bash
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin
POSTGRES_DB=tracking_db
COMBINER_CHECKPOINT=./checkpoint/combiner_model.pth
```

3. Start Services

```bash
docker compose up -d
```

4. Initialize Database Schema

```bash
docker exec -i ml_postgres psql -U admin -d tracking_db < schema.sql
```

5. Extract Embeddings & Populate Vector Search

```bash
python -m model_core.extract_embeddings
python -m model_core.populate_qdrant
```

6. Create Airflow Admin User

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
│   └── mlflow/          # MLflow database and artifact storage
├── model_core/          # Embedding extraction, combiner logic, vector DB indexing
├── services/
│   ├── api/             # FastAPI application
│   └── ui/              # Frontend web app static assets & Nginx config
├── docker-compose.yml   # Multi-container orchestration setup
└── schema.sql           # Initial database schema setup script
```
