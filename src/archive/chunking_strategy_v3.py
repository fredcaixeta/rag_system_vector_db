import re
from typing import Dict
from collections import defaultdict
from PyPDF2 import PdfReader
import json

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
    Processa o texto do diário criando chunks a cada 3 quebras de linha
    
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
            }
        ]
    }
    """
    date_pattern = re.compile(
        r'^(?P<day>\d{1,2})(?:st|nd|rd|th)? Day of (?P<month>[A-Za-z]+) (?P<year>18\d{2}) - (?P<title>.+)$'
    )
    
    result = {
        "metadata": {
            "total_days": 0,
            "total_chunks": 0,
            "chunks_per_day": defaultdict(int),
            "avg_chunks_per_day": 0.0
        },
        "chunks": []
    }
    
    current_date = None
    current_metadata = None
    chunk_number = 0
    buffer = []
    line_counter = 0
    
    def _create_chunk(result, buffer, current_date, current_metadata, chunk_number):
        chunk_text = "\n".join(buffer)
        palavras_maiusculas = re.findall(r'\b[A-Z]\w*\b', chunk_text)

        result["chunks"].append({
            "chunk_number": chunk_number,
            "chunk_text": chunk_text,
            "date": current_date,
            "day_metadata": {
                "full_date": current_metadata["full_date"],
                "title": current_metadata["title"],
                "palavras_maiusculas": palavras_maiusculas
            },
            "line_count": len(buffer),
            "word_count": len(chunk_text.split())
        })
        result["metadata"]["chunks_per_day"][current_date] += 1
        result["metadata"]["total_chunks"] += 1

    for line in text.split('\n'):
        line = line.strip()
        date_match = date_pattern.match(line)
        
        if date_match:
            # Processa buffer antes de nova data
            if buffer and current_date:
                _create_chunk(result, buffer, current_date, current_metadata, chunk_number)
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
            result["metadata"]["total_days"] += 1
            
            # Adiciona a linha de data como chunk separado
            result["chunks"].append({
                "chunk_number": chunk_number,
                "chunk_text": line,
                "date": current_date,
                "day_metadata": current_metadata,
                "line_count": 1,
                "word_count": len(line.split()),
                "is_date_chunk": True
            })
            result["metadata"]["chunks_per_day"][current_date] += 1
            result["metadata"]["total_chunks"] += 1
            chunk_number += 1
        else:
            if line:  # Ignora linhas vazias
                buffer.append(line)
                line_counter += 1
            
            # Cria chunk a cada 4 quebras de linha significativas
            if line_counter >= 4 and current_date:
                _create_chunk(result, buffer, current_date, current_metadata, chunk_number)
                chunk_number += 1
                buffer = []
                line_counter = 0

    # Processa o buffer final
    if buffer and current_date:
        _create_chunk(result, buffer, current_date, current_metadata, chunk_number)
        result["metadata"]["total_chunks"] += 1  # Atualiza o total de chunks
    
    # Calcula a média de chunks por dia
    total_days = result["metadata"]["total_days"]
    total_chunks = result["metadata"]["total_chunks"]
    result["metadata"]["avg_chunks_per_day"] = total_chunks / total_days if total_days > 0 else 0.0

    return result

if __name__ == '__main__':
    pdf_path = "data\dr_voss_diary.pdf"
    output_json = "diary_chunks_3breaks_v2.json"
    
    diary_text = extract_text_with_multiple_breaks(pdf_path)
    if diary_text:
        processed_data = process_diary_chunks(diary_text)
        
        # Exemplo de como acessar as informações
        print("Metadata:")
        print(f"  Total Days: {processed_data['metadata']['total_days']}")
        print(f"  Total Chunks: {processed_data['metadata']['total_chunks']}")
        print(f"  Avg Chunks per Day: {processed_data['metadata']['avg_chunks_per_day']:.2f}")
        
        print("\nChunks:")
        for chunk in processed_data["chunks"]:
            print(f"  Chunk Number: {chunk['chunk_number']}")
            print(f"  Date: {chunk['date']}")
            print(f"  Title: {chunk['day_metadata']['title']}")
            print(f"  Chunk Text: {chunk['chunk_text'][:50]}...")  # Printa apenas os primeiros 50 caracteres
            print(f"  Palavras Maiusculas: {chunk['day_metadata']['palavras_maiusculas']}")
            print("-" * 20)
            
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Dados salvos em: {output_json}")
    else:
        print("Não foi possível extrair o texto do PDF.")