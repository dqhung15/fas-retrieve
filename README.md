# fas-retrieve

docker compose up -d
docker exec -i ml_postgres psql -U admin -d tracking_db < schema.sql 
python -m model_core.extract_embeddings
python -m model_core.populate_qdrant

docker exec ml_airflow airflow users create \
    --username mlops \
    --firstname MLOps \
    --lastname Admin \
    --role Admin \
    --email mlops@example.com \
    --password admin
