import asyncio
import os
from typing import Dict, List, Optional

import pytest
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from scripts.milvus_db import ZillizClient
import src.groq_proxy as groq

# Carregue as variáveis de ambiente
load_dotenv()

class TestApp:
    def __init__(self):
        self.embedding_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-s")
        self.groq_client = groq.GroqProxyRestAPI()

    def generate_embedding(self, text: str) -> List[float]:
        """Gera embeddings normalizados para o texto de entrada."""
        return self.embedding_model.encode([text], normalize_embeddings=True)[0].tolist()

    async def query_document_logic(
        self, 
        question: str, 
        milvus_client: ZillizClient
    ) -> Dict[str, str]:
        """
        Lógica principal para buscar respostas no banco de dados vetorial.
        
        Args:
            question: Pergunta do usuário
            milvus_client: Cliente Zilliz/Milvus configurado
            
        Returns:
            Dicionário com resposta e contexto
        """
        try:
            # 1. Gerar embedding da pergunta
            question_embedding = self.generate_embedding(question)
            
            # 2. Buscar no Milvus
            search_results = milvus_client.search_vectors(
                collection_name=os.getenv("collection_name"),
                vector=question_embedding
            )
            
            if not search_results or not search_results.get("data"):
                return {"response": "Could not find relevant data.", "context": []}

            # 3. Processar resultados
            relevant_ids = [hit["id"] for hit in search_results["data"]]
            entities_data = milvus_client.get_entities_by_ids(
                collection_name=os.getenv("collection_name"),
                ids=relevant_ids
            )

            if not entities_data or not entities_data.get("data"):
                return {"response": "Data not found.", "context": []}

            # 4. Gerar resposta com Groq
            context = [entity["text"] for entity in entities_data["data"]]
            llm_answer = self.groq_client.generate_response(
                context=context, 
                question=question
            )
            
            return {
                "response": llm_answer,
                "context": context,
                "source_ids": relevant_ids
            }

        except Exception as e:
            error_msg = f"Error in query: {str(e)}"
            print(error_msg)
            return {"response": error_msg, "context": []}

def test_query_document_logic_real():
    """Testing."""
    # Configuração
    ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")
    ZILLIZ_CLUSTER_ID = os.getenv("ZILLIZ_CLUSTER_ID")
    COLLECTION_NAME = os.getenv("collection_name")

    if not all([ZILLIZ_API_KEY, ZILLIZ_CLUSTER_ID, COLLECTION_NAME]):
        pytest.skip("Env ERROR")

    # Inicializa clientes
    tester = TestApp()
    milvus_client = ZillizClient(
        api_key=ZILLIZ_API_KEY,
        cluster_id=ZILLIZ_CLUSTER_ID
    )

    # Teste
    test_question = "What is the currency of Veridia called?"
    result = asyncio.run(tester.query_document_logic(test_question, milvus_client))
    
    # Assertivas básicas
    assert "response" in result
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0
    
    print("\n🔍 Test:")
    print(f"Pergunta: {test_question}")
    print(f"Resposta: {result['response']}")
    # if result.get("context"):
    #     print(f"\nContexto usado ({result['context']}")
    #     for i, ctx in enumerate(result["context"][:3], 1):
    #         print(f"{i}. {ctx[:100]}...")

if __name__ == "__main__":
    test_query_document_logic_real()