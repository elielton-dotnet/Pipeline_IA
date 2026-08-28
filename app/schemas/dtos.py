from pydantic import BaseModel
from typing import Optional

class EvolucaoRequest(BaseModel):
    texto_clinico: str
    cenario: int            # 1 (Bruto), 2 (Limpo PT-BR), 3 (Limpo EN)
    modelo_alvo: str        # Ex: "emilyalsentzer/Bio_ClinicalBERT", "google/medgemma-1.5-4b-it"
    prompt_id: int          # 1, 2 ou 3 (Conforme literatura)

class InferenciasResponse(BaseModel):
    status: str
    predicao: str          
    confianca: float       
    modelo: str
    hardware_utilizado: str
    texto_processado: Optional[str] = None # Devolve o texto caso tenha sido traduzido