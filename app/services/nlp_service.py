import os
import torch
import torch.nn.functional as F
import gc
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import BitsAndBytesConfig

# Autenticação Hugging Face para liberar o download do MedGemma
os.environ["HF_TOKEN"] = "SEU_TOKEN_AQUI"  # Substitua pelo seu token real do Hugging Face

class NLPService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Estado do Modelo de Inferência
        self.current_model_name = None
        self.tokenizer = None
        self.model = None
        
        # Estado do Tradutor (MarianMT - Romance to English)
        self.translator_name = "Helsinki-NLP/opus-mt-roa-en"
        self.translator_tokenizer = None
        self.translator_model = None

    def limpar_memoria_gpu(self):
        """Força a limpeza da VRAM antes de carregar um novo modelo."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
        torch.cuda.empty_cache()
        gc.collect()
        print("[!] Memória da GPU (VRAM) limpa e liberada.")

    def carregar_tradutor_se_necessario(self):
        """Carrega o modelo de tradução offline sob demanda com proteção SafeTensors."""
        if self.translator_model is None:
            print("[*] Carregando Tradutor Offline (MarianMT)...")
            self.translator_tokenizer = AutoTokenizer.from_pretrained(self.translator_name)
            self.translator_model = AutoModelForSeq2SeqLM.from_pretrained(self.translator_name, use_safetensors=True).to(self.device)
            self.translator_model.eval()
            print("[*] Tradutor pronto!")

    def traduzir_pt_en(self, texto_pt: str) -> str:
        """Traduz o texto clínico limitando a 512 tokens de forma segura."""
        if not texto_pt:
            return ""
        entradas = self.translator_tokenizer(texto_pt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            saida_tokens = self.translator_model.generate(**entradas)
        texto_en = self.translator_tokenizer.decode(saida_tokens[0], skip_special_tokens=True)
        return texto_en

    def carregar_modelo_se_necessario(self, nome_modelo: str):
        """Lazy Loading: Gerencia a troca de modelos na GPU com segurança."""
        if self.current_model_name == nome_modelo:
            return 

        print(f"[*] Solicitada troca de modelo. Descarregando {self.current_model_name} e carregando {nome_modelo}...")
        self.limpar_memoria_gpu()
        
        self.current_model_name = nome_modelo
        self.tokenizer = AutoTokenizer.from_pretrained(nome_modelo)

        # SE FOR LLM (MedGemma): Carrega com Quantização de 4-bits
        if "gemma" in nome_modelo.lower():
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.model = AutoModelForCausalLM.from_pretrained(nome_modelo, quantization_config=quantization_config, device_map="auto")
        
        # SE FOR BERT (Encoders): Carrega normalmente
        else:
            self.model = AutoModel.from_pretrained(nome_modelo, use_safetensors=True).to(self.device)
            self.model.eval()
            
        print(f"[*] Modelo {nome_modelo} carregado com sucesso!")

    def classificar_texto(self, texto: str, prompt_id: int, nome_modelo: str):
        """Roteador Arquitetural blindado."""
        if not texto:
            return "Normal", 0.50
        if "gemma" in nome_modelo.lower():
            return self._classificar_llm(texto, prompt_id)
        else:
            return self._classificar_bert(texto)

    def _classificar_bert(self, texto: str):
        """Prompt 1: Matemática Tensorial e Softmax totalmente blindada."""
        labels = [
            "sinais de sepse, choque séptico ou infecção grave", 
            "paciente estável, sem infecção ou apenas inflamação"
        ]
        
        entradas_texto = self.tokenizer(texto, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        entradas_labels = self.tokenizer(labels, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs_texto = self.model(**entradas_texto)
            outputs_labels = self.model(**entradas_labels)
            
            emb_texto = outputs_texto.last_hidden_state[:, 0, :]
            emb_labels = outputs_labels.last_hidden_state[:, 0, :]
            
            emb_texto = F.normalize(emb_texto, p=2, dim=1)
            emb_labels = F.normalize(emb_labels, p=2, dim=1)
            
            similaridades = torch.matmul(emb_texto, emb_labels.t()).squeeze(0)
            probabilities = F.softmax(similaridades * 10, dim=0) 
            
            prob_sepse = probabilities[0].item()
            prob_normal = probabilities[1].item()
            
            if prob_sepse > prob_normal:
                return "Sepse", prob_sepse
            else:
                return "Normal", prob_normal

    def _classificar_llm(self, texto: str, prompt_id: int):
        """Prompts 2 e 3: Geração Textual Instrucional com tratamento rigoroso de índices."""
        if prompt_id == 2:
            prompt_texto = f"You are an ED doctor. Your task is to identify the following abnormal clinical signs and symptoms: Sepsis. Think step-by-step and provide your response in the following JSON format: {{'Sepsis': ['Yes or No', 'Concise justification']}}. Medical note: {texto}"
        else: 
            prompt_texto = f"You are an ED doctor. Your task is to identify if sepsis is present in the current admission. Think step-by-step and provide your response in the following JSON format: {{'Sepsis': ['Yes or No', 'Concise justification']}}. Note that 'Sepsis' is defined as suspicion or documentation of infection with evidence of organ dysfunction. Medical note: {texto}"

        entradas = self.tokenizer(prompt_texto, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            saida_ids = self.model.generate(
                **entradas, 
                max_new_tokens=100, 
                temperature=0.3, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        resposta_bruta = self.tokenizer.decode(saida_ids[0], skip_special_tokens=True)
        resposta_lower = resposta_bruta.lower()
        
        partes = resposta_lower.split("sepsis")
        ultima_parte = partes[-1] if len(partes) > 1 else resposta_lower

        if "'yes'" in resposta_lower or '"yes"' in resposta_lower or "yes" in ultima_parte:
             return "Sepse", 0.90 
        else:
             return "Normal", 0.90

nlp_service_instance = NLPService()