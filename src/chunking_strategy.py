import re
import json
from typing import Dict, List
from collections import defaultdict
from PyPDF2 import PdfReader

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
        "metadata": {
            "total_days": int,
            "total_chunks": int,
            "chunks_per_day": dict,
            "avg_chunks_per_day": float
        },
        "chunks": [
            {
                "chunk_number": int,
                "chunk_text": str,
                "date": str,
                "day_metadata": {
                    "full_date": str,
                    "title": str
                },
                "line_count": int,
                "word_count": int
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
                "title": date_match.group('title')
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
                
                # Cria chunk a cada 3 quebras de linha significativas
                if line_counter >= 3 and current_date:
                    _create_chunk(result, buffer, current_date, current_metadata, chunk_number)
                    chunk_number += 1
                    buffer = []
                    line_counter = 0
    
    # Processa qualquer conteúdo restante no buffer
    if buffer and current_date:
        _create_chunk(result, buffer, current_date, current_metadata, chunk_number)
    
    # Calcula média de chunks por dia
    if result["metadata"]["total_days"] > 0:
        result["metadata"]["avg_chunks_per_day"] = (
            result["metadata"]["total_chunks"] / result["metadata"]["total_days"]
        )
    
    return result

def _create_chunk(result: Dict, buffer: List[str], date: str, metadata: Dict, chunk_number: int):
    """Cria um novo chunk a partir do buffer"""
    chunk_text = '\n'.join(buffer)
    result["chunks"].append({
        "chunk_number": chunk_number,
        "chunk_text": chunk_text,
        "date": date,
        "day_metadata": metadata,
        "line_count": len(buffer),
        "word_count": len(chunk_text.split()),
        "is_date_chunk": False
    })
    result["metadata"]["chunks_per_day"][date] += 1
    result["metadata"]["total_chunks"] += 1

def save_chunks_to_json(pdf_path: str, output_file: str) -> Dict:
    """Processa o PDF e salva os chunks em JSON"""
    print(f"Processando: {pdf_path}")
    
    text = extract_text_with_multiple_breaks(pdf_path)
    if not text:
        return None
    
    diary_data = process_diary_chunks(text)
    
    print("\n📊 Estatísticas:")
    print(f"- Dias identificados: {diary_data['metadata']['total_days']}")
    print(f"- Total de chunks: {diary_data['metadata']['total_chunks']}")
    print(f"- Média de chunks por dia: {diary_data['metadata']['avg_chunks_per_day']:.1f}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(diary_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Dados salvos em: {output_file}")
    return diary_data

# Exemplo de uso
if __name__ == "__main__":
    pdf_file = "data\dr_voss_diary.pdf"
    output_json = "diary_chunks_3breaks.json"
    
    data = save_chunks_to_json(pdf_file, output_json)
    
    if data:
        print("\nExemplo de chunks:")
        for chunk in data["chunks"][:3] + data["chunks"][-2:]:
            prefix = "📅" if chunk.get("is_date_chunk", False) else "📦"
            print(f"{prefix} Chunk {chunk['chunk_number']} ({chunk['date']}):")
            print(f"   Linhas: {chunk['line_count']} | Palavras: {chunk['word_count']}")
            print(f"   Texto: {chunk['chunk_text'][:70]}...\n")