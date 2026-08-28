from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="API de PLN - Detecção de Sepse",
    description="Pipeline Multi-Modelo Zero-Shot (BERT e LLMs)"
)

# Injeta as nossas rotas
app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {"mensagem": "API Multi-Modelo operacional. Acesse /docs."}