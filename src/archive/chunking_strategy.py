import re
from PyPDF2 import PdfReader

def split_large_chunk(chunk: str, max_size: int = 800) -> list[str]:
    parts = []
    while len(chunk) > max_size:
        cut_index = chunk.rfind('.', 0, max_size)
        if cut_index == -1:
            cut_index = chunk.rfind(' ', 0, max_size)
        if cut_index == -1:
            cut_index = max_size  # Não achou ponto nem espaço, corta bruto
        parts.append(chunk[:cut_index+1].strip())
        chunk = chunk[cut_index+1:].strip()
    if chunk:
        parts.append(chunk)
    return parts

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except FileNotFoundError:
        print(f"Erro: File not found - {pdf_path}")
        return None
    return text

def chunk_diary_by_day_and_paragraph(diary_text: str) -> list[str]:
    chunks = []
    current_day_content = ""
    date_pattern = r"(\d{1,2})(?:st|nd|rd|th)? Day of ([A-Za-z]+) (18\d{2}) - ([A-Za-z\s]+)"

    for line in diary_text.splitlines():
        if re.match(date_pattern, line):
            # Encontrou um novo dia
            if current_day_content:
                # Divida o conteúdo do dia anterior por parágrafos (linhas não vazias)
                paragraphs = [p.strip() for p in current_day_content.strip().split('\n\n') if p.strip()]
                chunks.extend(paragraphs)
            current_day_content = line + "\n"  # Comece o conteúdo do novo dia
        else:
            current_day_content += line + "\n"

    # Processar o conteúdo do último dia
    if current_day_content:
        paragraphs = [p.strip() for p in current_day_content.strip().split('\n\n') if p.strip()]
        for paragraph in paragraphs:
            if len(paragraph) > 800:
                chunks.extend(split_large_chunk(paragraph))
            else:
                chunks.append(paragraph)
    
    return [chunk for chunk in chunks if chunk] # Remove empty ones 

if __name__ == "__main__":
    pdf_file = "data\dr_voss_diary.pdf"  # Substitua pelo caminho do seu arquivo PDF
    
    diary_text = extract_text_from_pdf(pdf_file)

    if diary_text:
        paragraph_chunks = chunk_diary_by_day_and_paragraph(diary_text)
        for i, chunk in enumerate(paragraph_chunks[:5]):
            pass
            print(f"--- Chunk {i+1} of {len(paragraph_chunks)} ---")
            print(f"  Tamanho: {len(chunk)} caracteres")
            print(f"  Conteúdo (primeiros 50 caracteres): {chunk[:50]}...")
            print("-" * 20)
        
    else:
        print("Não foi possível extrair o texto do PDF.")