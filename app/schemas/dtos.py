from pydantic import BaseModel

class EvolucaoRequest(BaseModel):
    texto_clinico: str
    cenario: int  # 1 para Bruto, 2 para Pré-processado

class InferenciasResponse(BaseModel):
    status: str
    predicao: str          # NOVO: Vai retornar "Sepse" ou "Normal"
    confianca: float       # NOVO: Porcentagem de certeza (ex: 0.85 para 85%)
    modelo: str
    hardware_utilizado: str