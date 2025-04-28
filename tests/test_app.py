# test_app.py
import asyncio
import os

import pytest
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Importe a classe ZillizClient
from scripts.milvus_db import ZillizClient

# Importing Groq Proxy
import src.groq_proxy as groq

# Carregue as variáveis de ambiente
load_dotenv()

# Função para gerar embeddings
def generate_embedding(text: str):
    # Opensource model for embedding - snowflake-arctic-embed-s
    model = SentenceTransformer("Snowflake/snowflake-arctic-embed-s")
    return model.encode([text], normalize_embeddings=True)[0].tolist()

# Função principal de consulta (similar à query_document do app.py)
async def query_document_logic(question: str, milvus_client: ZillizClient):
    """
    Main logic to receive a question and search answers in the vector database.
    """
    print(f"Pergunta de teste: {question}")

    # 1. Generate the question's embedding
    question_embedding = generate_embedding(question)
    
    # 2. Perform the search to the vector database - Milvus
    try:
        search_results = milvus_client.search_vectors(
            collection_name=os.getenv("collection_name"),
            vector=question_embedding
        )
        #print(f"Resultados da busca (teste real): {search_results}")

        # 3. Processar os resultados para formar uma resposta
        if search_results and search_results.get("data"):
                relevant_ids = [hit["id"] for hit in search_results["data"]]
                print(f"IDs relevantes encontrados: {relevant_ids}")

                if relevant_ids:
                    # 3. Buscar as entidades completas usando os IDs
                    entities_data = milvus_client.get_entities_by_ids(
                        collection_name=os.getenv("collection_name"),
                        ids=relevant_ids
                    )
                    print(f"Entidades recuperadas: {entities_data}")

                    if entities_data and entities_data.get("data"):
                        # 4. Use o LLM para gerar uma resposta baseada no contexto
                        context = [entity["text"] for entity in entities_data["data"]]
                        print(f"Contexto para LLM (Groq API REST): {context}")

                        # 4. Usar o GroqProxy para gerar uma resposta baseada no contexto
                        
                        groq_client = groq.GroqProxyRestAPI()
                        llm_answer = groq_client.generate_response(context=context, question=question)
                                                
                        return {"response": llm_answer, "context": context}
                    else:
                        return {"response": "Could not find relevant data within the document.", "context": []}
                else:
                    return {"response": "Não encontrei informações relevantes para sua pergunta.", "context": []}
        else:
            return {"response": "Não encontrei informações relevantes para sua pergunta.", "context": []}

    except Exception as e:
        raise Exception(f"Error while querying the database ao consultar o banco de vetores (teste real): {e}")

# Teste da lógica de consulta (chamando o serviço real do Milvus)
def test_query_document_logic_real():
    # Inicialize o ZillizClient com as credenciais reais
    ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")
    ZILLIZ_CLUSTER_ID = os.getenv("ZILLIZ_CLUSTER_ID")

    if not ZILLIZ_API_KEY or not ZILLIZ_CLUSTER_ID:
        pytest.skip("As variáveis de ambiente ZILLIZ_API_KEY e ZILLIZ_CLUSTER_ID não estão definidas. Pulando teste real.")

    milvus_client = ZillizClient(
        api_key=ZILLIZ_API_KEY,
        cluster_id=ZILLIZ_CLUSTER_ID
    )

    test_question = "What she did at the Gerlian Mountains?"
    result = asyncio.run(query_document_logic(test_question, milvus_client))
    
    print("Answer:", result["response"])

# Execute o teste real se o script for rodado diretamente
if __name__ == "__main__":
    test_query_document_logic_real()