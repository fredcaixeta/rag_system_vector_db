import os
import sys
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
from milvus_db import ZillizClient  # Importando a classe do arquivo separado

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.archive.chunking_strategy import chunk_diary_by_day_and_paragraph

# Carregar variáveis de ambiente
load_dotenv()

class PDFProcessor:
    def __init__(self):
        # Model - embeddings
        self.model = SentenceTransformer("Snowflake/snowflake-arctic-embed-s")
        self.embedding_dim = 384  # Dimensão dos embeddings do modelo Arctic-S
        self.collection_name = os.getenv("collection_name")
        # Client Milvus
        self.milvus_client = ZillizClient(
            api_key=os.getenv("ZILLIZ_API_KEY"),
            cluster_id=os.getenv("ZILLIZ_CLUSTER_ID"),
            region=os.getenv("ZILLIZ_REGION", "gcp-us-west1")
        )

    def extract_text_from_pdf(self, pdf_path) -> str:
        """Extrai texto de um arquivo PDF com metadados de página"""
        print(f"Extraindo texto de {pdf_path}...")
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except FileNotFoundError:
            print(f"Erro: File not found - {pdf_path}")
            return None
        return text


    def chunk_text(self, text) -> list[str]:
        """Divides Text into Chunks from Strategy set in - chunking_strategy.py."""
        print("Taking the chunks from chunking_strategy.py...")
        return chunk_diary_by_day_and_paragraph(text)

    def generate_embeddings(self, chunks) -> list[np.ndarray]:
        """Generates embeddings using model Snowflake Arctic"""
        print("Generating embeddings...")
        texts = [chunk for chunk in chunks]  # A função de chunking já retorna uma lista de strings
        return self.model.encode(texts, normalize_embeddings=True)

    def process_pdf(self, pdf_path):
        """Pipeline completo de processamento"""
        try:
            # 1. Extrair texto com metadados de página
            text = self.extract_text_from_pdf(pdf_path)
            
            # 2. Dividir em chunks com metadados
            chunks = self.chunk_text(text)
            total_chunks = len(chunks)
            print(f"Total de chunks = {total_chunks}")
            print("\n--- Primeiros 3 Chunks e seus tamanhos ---")
            for i, chunk in enumerate(chunks[:3]):
                print(f"Chunk {i+1}:")
                print(f"  Tamanho: {len(chunk)} caracteres")
                print(f"  Conteúdo (primeiros 50 caracteres): {chunk[:50]}...")
                print("-" * 20)
                
                
            # 3. Gerar embeddings
            embeddings = self.generate_embeddings(chunks)
            
            # 4. Preparar dados para o Milvus
            entities = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1):
                # Garante que todos os campos obrigatórios existam
                entity = {
                    "primary_key": int(idx),  # must be int
                    "vector": embedding.tolist(),
                    "text": chunk,
                }
            
                # Remove campos vazios ou None (opcional)
                entity = {k: v for k, v in entity.items() if v is not None}
                
                entities.append(entity)
                
                # 5. Armazenar no Milvus
                self.milvus_client.insert_vectors(self.collection_name, entities)
                
                print(f"✅ Process done! {idx} chunk sent.")
            
        except Exception as e:
            print(f"❌ Error within the process: {e}")
            raise

    def test_similarity(self, sentences):
        """Testa a similaridade entre frases"""
        embeddings = self.model.encode(sentences, normalize_embeddings=True)
        similarities = np.dot(embeddings, embeddings.T)
        print("Matriz de Similaridade:")
        print(similarities)
        return similarities

def main():
    processor = PDFProcessor()
    pdf_path = r"data\dr_voss_diary.pdf"
    #pdf_path = r"data\teste_pdf.pdf"
    # Processar PDF
    processor.process_pdf(pdf_path)
    
    # Teste de similaridade (opcional)
    test_sentences = [
        "That is a happy person",
        "That is a happy dog", 
        "That is a very happy person",
        "Today is a sunny day"
    ]
    #processor.test_similarity(test_sentences)

if __name__ == "__main__":
    main()