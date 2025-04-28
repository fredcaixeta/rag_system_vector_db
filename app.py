from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Adiciona o diretório raiz ao path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    
from scripts.milvus_db import ZillizClient
import src.groq_proxy as groq

# Carregue as variáveis de ambiente
load_dotenv()

app = FastAPI(
    title="Dr. Voss Diary RAG API",
    description="API for querying Dr. Elara Voss's research documents using RAG",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: str
    context: List[str]
    source_ids: List[str]
    success: bool

class RAGSystem:
    def __init__(self):
        # Modelo síncrono de embeddings
        self.embedding_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-s")
        self.groq_client = groq.GroqProxyRestAPI()
        self.milvus_client = self._initialize_milvus_client()

    def _initialize_milvus_client(self) -> ZillizClient:
        """Initialize and return Milvus/Zilliz client"""
        ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")
        ZILLIZ_CLUSTER_ID = os.getenv("ZILLIZ_CLUSTER_ID")
        
        if not ZILLIZ_API_KEY or not ZILLIZ_CLUSTER_ID:
            raise RuntimeError("Milvus/Zilliz credentials not configured")
            
        return ZillizClient(
            api_key=ZILLIZ_API_KEY,
            cluster_id=ZILLIZ_CLUSTER_ID
        )

    def generate_embedding(self, text: str) -> List[float]:
        """Generate normalized embeddings for input text (synchronous)"""
        return self.embedding_model.encode([text], normalize_embeddings=True)[0].tolist()

    def process_query(self, question: str) -> Dict:
        try:
            question_embedding = self.generate_embedding(question)
            
            search_results = self.milvus_client.search_vectors(
                collection_name=os.getenv("collection_name"),
                vector=question_embedding
            )
            
            if not search_results or not search_results.get("data"):
                return {
                    "response": "No relevant information found.",
                    "context": [],
                    "source_ids": [],
                    "success": False
                }

            # Conversão crucial dos IDs para string
            relevant_ids = [str(hit["id"]) for hit in search_results["data"]]
            
            entities_data = self.milvus_client.get_entities_by_ids(
                collection_name=os.getenv("collection_name"),
                ids=relevant_ids
            )

            if not entities_data or not entities_data.get("data"):
                return {
                    "response": "Could not retrieve document contents.",
                    "context": [],
                    "source_ids": [],
                    "success": False
                }

            context = [entity["text"] for entity in entities_data["data"]]
            llm_answer = self.groq_client.generate_response(
                context=context, 
                question=question
            )
            
            return {
                "response": llm_answer,
                "context": context,
                "source_ids": relevant_ids,  # Já convertidos
                "success": True
            }

        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "context": [],
                "source_ids": [],
                "success": False
            }

# Initialize the RAG system at startup
rag_system = RAGSystem()

@app.post("/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    """
    Synchronous endpoint to query the Dr. Voss diary documents
    
    Parameters:
    - question: The question about Veridia's world
    
    Returns:
    - response: Generated answer
    - context: Relevant chunks used
    - source_ids: IDs of source documents
    - success: Whether the operation succeeded
    """
    result = rag_system.process_query(request.question)
    
    if not result["success"]:
        raise HTTPException(
            status_code=404,
            detail=result["response"]
        )
    
    return result

@app.get("/health")
def health_check():
    """Health check endpoint (synchronous)"""
    return {"status": "healthy", "services": ["milvus", "embedding", "llm"]}

@app.on_event("startup")
def startup_event():
    """Initialize components when app starts (synchronous)"""
    try:
        test_embedding = rag_system.generate_embedding("test")
        assert len(test_embedding) == 384
    except Exception as e:
        raise RuntimeError(f"Embedding model initialization failed: {str(e)}")