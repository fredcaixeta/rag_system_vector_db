import os

GROQ_API_KEY = os.getenv("groq_key")
# src/groq_proxy.py
import os
import requests
import json

class GroqProxyRestAPI:
    def __init__(self, api_key=None, model_name="llama3-8b-8192"):
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY não encontrado nas variáveis de ambiente.")
        self.base_url = "https://api.groq.com/openai/v1"
        self.model_name = model_name
        
    def eval(self, context):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": f"You are a research assistant. Use the following context to answer the question: {context}"}
            ],
            "top_p": 1,
            "stream": False,
            "stop": None
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            response_json = response.json()
            return response_json['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            print(f"Erro ao chamar a API REST da Groq: {e}")
            if response is not None:
                print(f"Status Code: {response.status_code}")
                print(f"Response Body: {response.text}")
            return "Não consegui gerar uma resposta usando o LLM (API REST)."

    def generate_response(self, question: str, context: str, max_tokens: int = 2000, temperature: float = 0.3):
        """Gera uma resposta usando a API REST da Groq."""
        print(f"Context related: {context}")
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": f"You are a research assistant. Use the following context to answer the question: {context}"},
                {"role": "system", "content": "If the query is not related to context, answer 'Could not find relevant data within the document'."},
                {"role": "user", "content": f"User query: {question}"}
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "top_p": 1,
            "stream": False,
            "stop": None
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            response_json = response.json()
            return response_json['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            print(f"Erro ao chamar a API REST da Groq: {e}")
            if response is not None:
                print(f"Status Code: {response.status_code}")
                print(f"Response Body: {response.text}")
            return "Não consegui gerar uma resposta usando o LLM (API REST)."