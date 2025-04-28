# scripts/prepare_data.py
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
from milvus_db import ZillizClient  # Importando a classe do arquivo separado
import uuid
from src.archive.chunking_strategy import chunk_diary_by_day_and_paragraph  # Bringing up Chunking Strategy
# Carregar variáveis de ambiente
load_dotenv()

class PDFProcessor:
    def __init__(self):
        # Modelo de embeddings
        self.model = SentenceTransformer("Snowflake/snowflake-arctic-embed-s")
        self.embedding_dim = 384  # Dimensão dos embeddings do modelo Arctic-S
        
        # Cliente Milvus
        self.milvus_client = ZillizClient(
            api_key=os.getenv("ZILLIZ_API_KEY"),
            cluster_id=os.getenv("ZILLIZ_CLUSTER_ID"),
            region=os.getenv("ZILLIZ_REGION", "gcp-us-west1")
        )

    def extract_text_from_pdf(self, pdf_path):
        """Extrai texto de um arquivo PDF com metadados de página"""
        print(f"Extraindo texto de {pdf_path}...")
        reader = PdfReader(pdf_path)
        text = ""
        page_metadata = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            text += page_text
            page_metadata.extend([page_num] * len(page_text.split()))  # Aproximação
            
        return text, page_metadata

    def chunk_text(self, text, page_metadata, chunk_size=1000, chunk_overlap=200):
        """Divide o texto em pedaços menores, preservando metadados de página"""
        print("Dividindo texto em chunks...")
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Criar lista de palavras com metadados
        words = text.split()
        word_page_pairs = list(zip(words, page_metadata[:len(words)]))
        
        # Reconstruir texto com marcadores de página aproximados
        chunks = []
        current_chunk = []
        current_chunk_pages = set()
        
        for i in range(0, len(word_page_pairs), chunk_size - chunk_overlap):
            chunk_words = word_page_pairs[i:i + chunk_size]
            chunk_text = ' '.join([word for word, page in chunk_words])
            chunk_pages = {page for word, page in chunk_words}
            
            chunks.append({
                "text": chunk_text,
                "pages": sorted(chunk_pages),
                "start_word": i,
                "end_word": min(i + chunk_size, len(word_page_pairs))
            })
        
        return chunks

    def generate_embeddings(self, chunks):
        """Gera embeddings usando o modelo Snowflake Arctic"""
        print("Gerando embeddings...")
        texts = [chunk["text"] for chunk in chunks]
        return self.model.encode(texts, normalize_embeddings=True)

    def process_pdf(self, pdf_path, collection_name="dr_voss"):
        """Pipeline completo de processamento"""
        try:
            # 1. Extrair texto com metadados de página
            text, page_metadata = self.extract_text_from_pdf(pdf_path)
            
            # 2. Dividir em chunks com metadados
            chunks = self.chunk_text(text, page_metadata)
            
            # 3. Gerar embeddings
            embeddings = self.generate_embeddings(chunks)
            
            # 4. Preparar dados para o Milvus
            entities = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1):
                # Garante que todos os campos obrigatórios existam
                entity = {
                    "primary_key": int(idx),  # Força conversão para inteiro
                    "vector": embedding.tolist(),
                    "text": chunk.get("text", ""),  # Valor padrão se não existir
                }
            
                # Remove campos vazios ou None (opcional)
                entity = {k: v for k, v in entity.items() if v is not None}
                
                entities.append(entity)
                
                # 5. Armazenar no Milvus
                self.milvus_client.insert_vectors(collection_name, entities)
                
                print(f"✅ Process done! {len(chunks)} chunks sent.")
            
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