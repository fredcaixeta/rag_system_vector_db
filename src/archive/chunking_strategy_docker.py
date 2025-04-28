import re
from typing import Dict, List
from collections import defaultdict
from PyPDF2 import PdfReader
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema, IndexType
import numpy as np

import sys
import os



from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Snowflake/snowflake-arctic-embed-s")
embedding_dim = 384  # Dimensão dos embeddings do modelo Arctic-S

def generate_embeddings(chunks) -> list[np.ndarray]:
    """Generates embeddings using model Snowflake Arctic"""
    print("Generating embeddings...")
    texts = [chunk for chunk in chunks]  # A função de chunking já retorna uma lista de strings
    return model.encode(texts, normalize_embeddings=True)

def extract_text_with_multiple_breaks(pdf_path: str) -> str:
    """Extrai texto do PDF preservando múltiplas quebras de linha"""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Erro ao extrair texto: {e}")
        return None
    return text

def process_diary_chunks(text: str) -> Dict:
    """
    Processa o texto do diário, criando chunks a cada 4 quebras de linha.

    Retorna:
    {
        "chunks": [
            {
                "chunk_number": int,
                "chunk_text": str,
                "day_metadata": {
                    "full_date": str,
                    "title": str,
                    "palavras_maiusculas": list
                }
            },
            ...
        ]
    }
    """
    date_pattern = re.compile(
        r'^(?P<day>\d{1,2})(?:st|nd|rd|th)? Day of (?P<month>[A-Za-z]+) (?P<year>18\d{2}) - (?P<title>.+)$'
    )
    
    result = {"chunks": []}
    current_date = None
    current_metadata = None
    chunk_number = 0
    buffer = []
    line_counter = 0
    
    for line in text.split('\n'):
        line = line.strip()
        date_match = date_pattern.match(line)
        
        if date_match:
            # Processa buffer antes de nova data
            if buffer and current_date:
                chunk_text = "\n".join(buffer)
                palavras_maiusculas = re.findall(r'\b[A-Z]\w*\b', chunk_text)
                result["chunks"].append({
                    "chunk_number": chunk_number,
                    "chunk_text": chunk_text,
                    "day_metadata": current_metadata,
                    "palavras_maiusculas": palavras_maiusculas
                })
                chunk_number += 1
                buffer = []
                line_counter = 0
            
            # Nova data encontrada
            current_date = line
            current_metadata = {
                "full_date": line,
                "title": date_match.group('title'),
                "palavras_maiusculas": re.findall(r'\b[A-Z]\w*\b', line)
            }
        else:
            if line:
                buffer.append(line)
                line_counter += 1
            
            if line_counter >= 4 and current_date: # Breaking at 4th \n line
                chunk_text = "\n".join(buffer)
                palavras_maiusculas = re.findall(r'\b[A-Z]\w*\b', chunk_text)
                result["chunks"].append({
                    "chunk_number": chunk_number,
                    "chunk_text": chunk_text,
                    "day_metadata": current_metadata,
                    "palavras_maiusculas": palavras_maiusculas
                })
                chunk_number += 1
                buffer = []
                line_counter = 0

    # Processa o buffer final
    if buffer and current_date:
        chunk_text = "\n".join(buffer)
        palavras_maiusculas = re.findall(r'\b[A-Z]\w*\b', chunk_text)
        result["chunks"].append({
            "chunk_number": chunk_number,
            "chunk_text": chunk_text,
            "day_metadata": current_metadata,
            "palavras_maiusculas": palavras_maiusculas
        })

    return result

def insert_data_into_milvus(collection_name: str, chunks: List[dict], embeddings: List[List[float]]):
    """
    Versão simplificada da inserção de dados no Milvus.
    Mantém apenas os campos essenciais: chunk_text, content_vector e palavras_maiusculas.
    """
    milvus_client = MilvusClient(uri=os.getenv("collection_uri"), token=os.getenv("MILVUS_token"))
    
    # Verifica e prepara a coleção
    try:
        milvus_client.drop_collection(collection_name)
    except:
        pass  # Coleção não existia
    
    # Schema simplificado
    fields = [
        FieldSchema(name="entry_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="content_vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="palavras_maiusculas", dtype=DataType.JSON)
    ]
    schema = CollectionSchema(fields=fields, description="Simplified Dr Voss Diary Schema")
    
    milvus_client.create_collection(collection_name, schema=schema)
    
    # Prepara os dados para inserção (apenas campos essenciais)
    entities = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        entities.append({
            "entry_id": f"chunk_{idx}",  # ID simples sequencial
            "content_vector": embedding,
            "chunk_text": chunk["chunk_text"],
            "palavras_maiusculas": chunk["palavras_maiusculas"]
        })
    
    # Insere os dados
    milvus_client.insert(collection_name, entities)
    
    # Cria índice (obrigatório para busca vetorial)
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    milvus_client.create_index(
        collection_name,
        field_name="content_vector",
        index_params=index_params
    )
    
    print(f"✅ Dados inseridos: {len(entities)} chunks (campos reduzidos)")
    return milvus_client


if __name__ == '__main__':
    
    # 1) Data Extraction
    pdf_path = "data\dr_voss_diary.pdf"
    diary_text = extract_text_with_multiple_breaks(pdf_path)

    if not diary_text:
        print("Error")
        exit(1)
    
    # 2) Chunk Processing
    processed_data = process_diary_chunks(diary_text)
    print(f"📊 Chunks processados: {len(processed_data['chunks'])}")
    
    # 3) Embeddings Generation
    chunks_text = [chunk["chunk_text"] for chunk in processed_data["chunks"]]
    embeddings = generate_embeddings(chunks_text)
    
    # 4) Milvus Insert
    milvus_collection_name = "dr_voss_diary_v2"
    milvus_client = insert_data_into_milvus(
        milvus_collection_name,
        processed_data["chunks"],
        embeddings
    )
    
    # 5) Test
    test_question = "Qual era o maior interesse na viagem?"
    question_embedding = generate_embeddings([test_question])[0]
    
    search_results = milvus_client.search(
        collection_name=milvus_collection_name,
        data=[question_embedding],
        limit=3,
        output_fields=["chunk_text", "entry_date", "word_count"]
    )
    
    print("\n🔍 Resultados da busca de teste:")
    for idx, hit in enumerate(search_results[0]):
        print(f"{idx+1}. Similaridade: {hit.score:.3f}")
        print(f"   Data: {hit.entity['entry_date']}")
        print(f"   Texto: {hit.entity['chunk_text'][:100]}...\n")