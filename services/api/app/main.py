from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api.app.model_loader import lifespan
from services.api.app.routes import router

app = FastAPI(
  title="Fashion Retrieval API",
  version="1.0.0",
  lifespan=lifespan
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)

@app.get("/")
async def root():
  return {"message": "Welcome to the Fashion Retrieval API. Visit /docs to test the endpoints."}

app.include_router(router)