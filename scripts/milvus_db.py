import requests
import json
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import numpy as np

# from pymilvus import MilvusClient
# client = MilvusClient("./milvus_demo.db")

# Carregar variáveis de ambiente
load_dotenv()

class ZillizClient:
    def __init__(self, api_key: str, cluster_id: str, region: str = "gcp-us-west1"):
        self.base_url = f"https://{cluster_id}.serverless.{region}.cloud.zilliz.com/v2"
        self.headers = {
            "accept": "application/json",
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json"
        }
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url}/{endpoint}"
        response = requests.request(
            method=method,
            url=url,
            headers=self.headers,
            json=data if data else {}
        )
        response.raise_for_status()
        
        print(response.json())
        return response.json()
    
    def list_collections(self) -> List[Dict]:
        """Lista todas as coleções no cluster"""
        return self._make_request("POST", "vectordb/collections/list")
    
    def create_collection(self, collection_name: str, dimension: int) -> Dict:
        """Cria uma nova coleção"""
        data = {
            "collectionName": collection_name,
            "dimension": dimension,
            "metricType": "COSINE",
            "primaryField": "id",
            "vectorField": "vector"
        }
        return self._make_request("POST", "vectordb/collections/create", data)
    
    def query_entities(self, collection_name: str, filter: str = "", output_fields: List[str] = ["*"], limit: int = 10):
        """Consulta entidades com filtro"""
        payload = {
            "collectionName": collection_name,
            "filter": filter,
            "outputFields": output_fields,
            "limit": limit
        }
        return self._make_request("POST", "vectordb/entities/query", payload)
    
    def insert_vectors(self, collection_name: str, data: List[Dict]) -> Dict:
        #print(f"Data being sent to VDB: {data}")
        """Insere vetores na coleção"""
        payload = {
            "collectionName": collection_name,
            "data": data
        }
        return self._make_request("POST", "vectordb/entities/insert", payload)
    
    def get_collection_stats(self, collection_name: str):
        """Obtém estatísticas da coleção"""
        payload = {
            "collectionName": collection_name
        }
        return self._make_request("POST", "vectordb/collections/describe", payload)
    
    def get_all_entities(self, collection_name: str, batch_size: int = 100):
        """Obtém todas as entidades de uma coleção (em lotes)"""
        all_entities = []
        offset = 0
        
        while True:
            payload = {
                "collectionName": collection_name,
                "outputFields": ["*"],
                "limit": batch_size,
                "offset": offset
            }
            result = self._make_request("POST", "vectordb/entities/query", payload)
            
            if not result.get("data"):
                break
                
            all_entities.extend(result["data"])
            offset += batch_size
            
        return all_entities
    
    def get_entities_by_ids(self, collection_name: str, ids: List[int]) -> Dict:
        """Obtém entidades por seus IDs."""
        payload = {
            "collectionName": collection_name,
            "id": ids
        }
        return self._make_request("POST", "vectordb/entities/get", payload)
    
    def search_vectors(self, collection_name: str, vector: List[float], limit: int = 5) -> Dict:
        """Realiza uma busca por similaridade"""
        payload = {
            "collectionName": collection_name,
            "data": [vector],
            "limit": 1
        }
        return self._make_request("POST", "vectordb/entities/search", payload)

# Exemplo de uso:
if __name__ == "__main__":
    API_KEY = os.getenv("ZILLIZ_API_KEY")
    CLUSTER_ID = os.getenv("ZILLIZ_CLUSTER_ID")
    COLL_NAME = os.getenv("collection_name")
    
    client = ZillizClient(api_key=API_KEY, cluster_id=CLUSTER_ID)
    
    # 1. Listar coleções existentes
    print("\n=== Coleções existentes ===")
    collections = client.list_collections()
    print(json.dumps(collections, indent=2))
    
    if collections.get("data"):
        # 2. Estatísticas da primeira coleção (para exemplo)
        for coll in collections["data"]:
            if coll == COLL_NAME:
                voss_coll = collections["data"]
                voss_coll = voss_coll[0]
                
        print(f"\n=== Estatísticas da coleção {voss_coll} ===")
        stats = client.get_collection_stats(voss_coll)
        print(json.dumps(stats, indent=2))
        
        # 3. Amostra de entradas (primeiras 5)
        print(f"\n=== Amostra de entradas em {voss_coll} ===")
        sample_data = client.query_entities(voss_coll, limit=5)
        print(json.dumps(sample_data, indent=2))
        
        # 4. Contagem total de entidades
        print(f"\n=== Contagem total em {voss_coll} ===")
        all_entities = client.get_all_entities(voss_coll)
        print(f"Total de entidades: {len(all_entities)}")
    else:
        print("\nNenhuma coleção encontrada no cluster.")
    # 2. Criar uma nova coleção (se necessário)
    # client.create_collection("dr_voss", 384)  # 384 é a dimensão do Snowflake Arctic Embed
    
    # 3. Inserir dados (exemplo)
    # example_data = [{
    #     "id": 1,
    #     "vector": [0.1]*384,  # Substitua por um embedding real
    #     "text": "Texto de exemplo",
    #     "page_number": 1
    # }]
    # insert_result = client.insert_vectors("dr_voss", example_data)
    # print("Resultado da inserção:", insert_result)
    
    # 4. Buscar similares (exemplo)
    # query_vector = [0.1]*384  # Substitua por um embedding real
    # search_results = client.search_vectors("dr_voss", query_vector)
    # print("Resultados da busca:", search_results)