# 🏥 Pipeline IA - Detecção de Sepse (Zero-Shot)

Este repositório contém o microsserviço de Inteligência Artificial responsável pelo processamento de linguagem natural (PLN) das evoluções clínicas. O projeto expõe uma API desenvolvida em **FastAPI** que utiliza o modelo **Bio_ClinicalBERT** para realizar inferências em tempo real utilizando a abordagem *Zero-Shot Classification*.

Este módulo atua como o motor analítico da arquitetura, recebendo os textos normalizados do backend (Projeto SepsisNlp feito em C#) e retornando o distanciamento semântico (Similaridade de Cosseno) entre o quadro do paciente e as hipóteses de risco (Sepse) ou controle.

## 🚀 Tecnologias Utilizadas

*   **Python 3**
*   **FastAPI** (Framework Web assíncrono e de alta performance)
*   **PyTorch** (Processamento tensorial com suporte a aceleração via GPU/CUDA)
*   **Transformers (Hugging Face)** (Integração com o modelo `emilyalsentzer/Bio_ClinicalBERT`)
*   **Uvicorn** (Servidor ASGI)

## 📁 Arquitetura (Clean Architecture Simulada)

O projeto está estruturado para separar responsabilidades, mantendo o código limpo e escalável:

```text
Pipeline_IA/
├── app/
│   ├── api/
│   │   └── routes.py         # Endpoints da API (Controllers)
│   ├── schemas/
│   │   └── dtos.py           # Contratos de entrada/saída (Pydantic Models)
│   ├── services/
│   │   └── nlp_service.py    # Regras de negócio, carregamento do modelo e matemática tensorial
│   ├── main.py               # Ponto de entrada da aplicação e ciclo de vida
│   └── .gitignore            # Regras de exclusão do Git
└── README.md