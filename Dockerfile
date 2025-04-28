# Usar uma imagem base com Python
FROM python:3.11-slim

# Atualizar pip e instalar Milvus Lite
RUN pip install --upgrade pip && pip install pymilvus milvus-lite

# Criar uma pasta para o app
WORKDIR /app

# Copiar um script para iniciar Milvus Lite (vamos criar logo abaixo)
COPY start_server.py .

# Expõe a porta padrão do Milvus (19530)
EXPOSE 19530

# Rodar o servidor quando o container iniciar
CMD ["python", "start_server.py"]
