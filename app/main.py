from fastapi import FastAPI
from app.api.routes import router
from app.services.nlp_service import nlp_service_instance

# Inicializa a aplicação
app = FastAPI(
    title="API de PLN - Detecção de Sepse",
    description="Pipeline de inferência usando Bio_ClinicalBERT"
)

# Ciclo de vida: O que rodar quando a API "subir"
@app.on_event("startup")
def startup_event():
    # Dispara o carregamento do modelo na RTX 4060
    nlp_service_instance.carregar_modelo()

# Injeta as nossas rotas (Controllers)
app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {"mensagem": "API operacional. Acesse /docs para visualizar o Swagger."}