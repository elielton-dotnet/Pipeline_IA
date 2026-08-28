from fastapi import APIRouter
from app.schemas.dtos import EvolucaoRequest, InferenciasResponse
from app.services.nlp_service import nlp_service_instance

router = APIRouter()

@router.post("/processar-evolucao", response_model=InferenciasResponse)
def processar_evolucao(request: EvolucaoRequest):
    
    # 1. Garante que o modelo correto está carregado na VRAM (Lazy Loading)
    nlp_service_instance.carregar_modelo_se_necessario(request.modelo_alvo)

    # 2. Se for cenário 3, executa a tradução offline
    texto_final = request.texto_clinico
    if request.cenario == 3:
        nlp_service_instance.carregar_tradutor_se_necessario()
        texto_final = nlp_service_instance.traduzir_pt_en(request.texto_clinico)

    # 3. Chama a matemática de IA passando o ID do Prompt da literatura
    predicao, confianca = nlp_service_instance.classificar_texto(
        texto=texto_final, 
        prompt_id=request.prompt_id,
        nome_modelo=request.modelo_alvo
    )
    
    print("\n" + "=" * 50)
    print(f"MODELO: {request.modelo_alvo} | PROMPT: {request.prompt_id} | CENÁRIO: {request.cenario}")
    print(f"RESULTADO DA IA: {predicao} (Confiança: {confianca:.2%})")
    print("=" * 50 + "\n")
    
    return InferenciasResponse(
        status="Sucesso",
        predicao=predicao,
        confianca=confianca,
        modelo=request.modelo_alvo,
        hardware_utilizado=nlp_service_instance.device,
        texto_processado=texto_final if request.cenario == 3 else None
    )