import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class NLPService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.tokenizer = None
        self.model = None

    def carregar_modelo(self):
        print(f"[*] Inicializando NLP Service. Hardware ativado: {self.device.upper()}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, use_safetensors=True).to(self.device)
        
        # Coloca o modelo em modo de avaliação (desliga os gradientes de treino)
        self.model.eval()
        print("[*] Modelo carregado e pronto para inferências!")

    def classificar_zero_shot(self, texto: str):
        # Nossas duas "âncoras" de conhecimento
        labels = ["sinais de sepse, choque séptico ou infecção grave", "paciente estável, sem infecção ou apenas inflamação"]
        
        # 1. Transformar textos em tokens matemáticos (limitado a 512 tokens para não estourar a memória)
        entradas_texto = self.tokenizer(texto, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        entradas_labels = self.tokenizer(labels, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Desliga o motor de aprendizado (economiza muita memória da GPU)
        with torch.no_grad():
            # 2. Extrair o conhecimento profundo (Pegamos o token [CLS] que resume a frase)
            emb_texto = self.model(**entradas_texto).last_hidden_state[:, 0, :]
            emb_labels = self.model(**entradas_labels).last_hidden_state[:, 0, :]
            
            # 3. Normalizar vetores
            emb_texto = F.normalize(emb_texto, p=2, dim=1)
            emb_labels = F.normalize(emb_labels, p=2, dim=1)
            
            # 4. Calcular Similaridade de Cosseno (A distância entre o texto e as âncoras)
            similaridades = torch.mm(emb_texto, emb_labels.transpose(0, 1)).squeeze(0)
            
            # 5. Transformar em Porcentagem (Softmax)
            # O * 10 é um truque acadêmico (temperature scaling) para afastar as probabilidades e dar mais contraste
            probabilidades = F.softmax(similaridades * 10, dim=0) 
            
            prob_sepse = probabilidades[0].item()
            prob_normal = probabilidades[1].item()
            
            if prob_sepse > prob_normal:
                return "Sepse", prob_sepse
            else:
                return "Normal", prob_normal

# Instância global
nlp_service_instance = NLPService()