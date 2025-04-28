import os
import sys
import json
from typing import List, Dict
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

# Adiciona o diretório raiz ao path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

def generate_embedding(text: str):
    """Gera o embedding de um texto usando o modelo Sentence Transformer."""

    model = SentenceTransformer("Snowflake/snowflake-arctic-embed-s")
    return model.encode([text], normalize_embeddings=True)[0].tolist()

def parse_qa_files(questions_file: str, answers_file: str) -> List[Dict[str, str]]:
    """
    Parseia os arquivos de perguntas e respostas, retornando uma lista de dicionários.

    Args:
        questions_file: Caminho para o arquivo com as perguntas.
        answers_file: Caminho para o arquivo com as respostas.

    Returns:
        Uma lista de dicionários, onde cada dicionário contém as chaves "question" e "expected_answer".
    """
    qa_pairs = []
    try:
        with open(questions_file, "r") as q_file, open(answers_file, "r") as a_file:
            questions = q_file.readlines()
            answers = a_file.readlines()

            # Garante que há um número igual de perguntas e respostas
            min_len = min(len(questions), len(answers))
            for i in range(min_len):
                qa_pairs.append({
                    "question": questions[i].strip(),
                    "expected_answer": answers[i].strip()
                })
    except FileNotFoundError as e:
        print(f"Erro ao carregar arquivos: {e}")
    return qa_pairs

def evaluate_rag_with_groq(qa_pairs: List[Dict[str, str]], groq_client: groq.GroqProxyRestAPI, milvus_client: ZillizClient) -> List[Dict[str, any]]:
    """
    Avalia o RAG pipeline usando o Groq para avaliar as respostas do LLM.

    Args:
        qa_pairs: Lista de dicionários contendo perguntas e respostas esperadas.
        groq_client: Instância do cliente GroqProxy.
         milvus_client: Instância do cliente Zilliz para interagir com o banco de dados vetorial.

    Returns:
        Uma lista de dicionários contendo os resultados da avaliação, incluindo a nota do Groq.
    """

    evaluation_results = []
    for qa in qa_pairs:
        question = qa["question"]
        expected_answer = qa["expected_answer"]

        # 1. Gerar embedding da pergunta
        question_embedding = generate_embedding(question)

        # 2. Buscar no banco de dados vetorial
        search_results = milvus_client.search_vectors(
            collection_name=os.getenv("collection_name"),  # Use a variável de ambiente
            vector=question_embedding,
        )

        if search_results and search_results.get("data"):
            relevant_ids = [hit["id"] for hit in search_results["data"]]
            if relevant_ids:
                # 3. Buscar entidades completas por IDs
                entities_data = milvus_client.get_entities_by_ids(
                    collection_name=os.getenv("collection_name"),  # Use a variável de ambiente
                    ids=relevant_ids
                )
                if entities_data and entities_data.get("data"):
                    context = [entity["text"] for entity in entities_data["data"]]
                    # 4. Obter resposta do LLM                    
                    predicted_answer = groq_client.generate_response(question, context)

                else:
                    predicted_answer = "Could not find relevant data within the document."
            else:
                predicted_answer = "Não encontrei informações relevantes para sua pergunta."
        else:
            predicted_answer = "Não encontrei informações relevantes para sua pergunta."

        # 5. Evaluation with LLM
        evaluation_prompt = f"""
        You are a question and answer system response evaluator.
        Given the question: "{question}", the expected answer: "{expected_answer}" and the system's answer: "{predicted_answer}",
        assign a grade from 0 to 1, where 1 indicates that the system's answer is perfectly aligned with the expected answer and 0 indicates that there is no alignment at all.

        Grade (0-1):
        """
        groq_evaluation = groq_client.eval(context=evaluation_prompt) # Pass an empty list as context

        evaluation_results.append({
            "question": question,
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
            "groq_evaluation": groq_evaluation
        })

    return evaluation_results

if __name__ == "__main__":
    # Substitua pelos caminhos corretos dos seus arquivos e suas chaves de API
    questions_file = r"data\questions.txt"
    answers_file = r"data\answers.txt"

    # Inicialize o cliente GroqProxy
    groq_client = groq.GroqProxyRestAPI()
    ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")
    ZILLIZ_CLUSTER_ID = os.getenv("ZILLIZ_CLUSTER_ID")

    # Inicialize o cliente Zilliz
    milvus_client = ZillizClient(
        api_key=ZILLIZ_API_KEY,
        cluster_id=ZILLIZ_CLUSTER_ID
    )

    # 1. Parsear os arquivos de perguntas e respostas
    qa_pairs = parse_qa_files(questions_file, answers_file)

    # 2. Avaliar o pipeline RAG
    evaluation_results = evaluate_rag_with_groq(qa_pairs, groq_client, milvus_client)

    # 3. Imprimir os resultados em JSON
    print(json.dumps(evaluation_results, indent=4, ensure_ascii=False))
    with open("evaluation_results.json", "w", encoding="utf-8") as json_file:
        json.dump(evaluation_results, json_file, indent=4, ensure_ascii=False)