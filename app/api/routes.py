from fastapi import APIRouter
from app.schemas.dtos import EvolucaoRequest, InferenciasResponse
from app.services.nlp_service import nlp_service_instance

router = APIRouter()

@router.post("/processar-evolucao", response_model=InferenciasResponse)
def processar_evolucao(request: EvolucaoRequest):
    
    # Chama a nossa matemática de IA
    predicao, confianca = nlp_service_instance.classificar_zero_shot(request.texto_clinico)
    
    print("\n" + "=" * 50)
    print(f"CENÁRIO: {request.cenario}")
    print(f"RESULTADO DA IA: {predicao} (Confiança: {confianca:.2%})")
    print("=" * 50 + "\n")
    
    return InferenciasResponse(
        status="Sucesso",
        predicao=predicao,
        confianca=confianca,
        modelo=nlp_service_instance.model_name,
        hardware_utilizado=nlp_service_instance.device
    )