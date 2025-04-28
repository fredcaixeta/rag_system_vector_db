## Dr. Voss Diary RAG System

### 1. **Installation & Setup Instructions**  

- **Environment Setup:**  
  - Python 3.10+
    - Groq Key - Acquire one for free at the Developer sector at https://console.groq.com/keys

- **Dependency Installation:**  
  - Provide installation commands (e.g., `pip install -r requirements.txt`).  

- **Running the Scripts & Application:**  
  - The script - `prepare_data.py` - creates the environment at the could vector database.
    
    python scripts/prepare_data.py


  - The script - `eval.py` - evaluates the full RAG system - check the output at 'data\evaluation_results.json'

    Evaluate the RAG system by running:

    python scripts/eval.py

    Output will be saved at:

    data/evaluation_results.json


- **FAST API Server Documentation:**

    - The FastAPI service provides the following endpoints:

### `POST /query`
Main endpoint for querying the Dr. Voss diary documents

**Request:**
```json
{
  "question": "What is the currency of Veridia called?"
}
```

### 2. **Technical Discussion:**  
   
   - **Model Selection:**
     - Choice of embedding model - opensource provider - low CPU necessity
     - Choice of LLM - using groq as proxy and its RESTAPI was necessary due to the 1000+ characters added to the context of the LLM. Important to create fast dozens of tests and fast evaluations - not relying on local performance.
   - **Data Processing:**
     - Document parsing - with PdfReader python library
     - Chunking strategy - separating into chunks the days of the diary
   - **Retrieval System:**
     - Vector database design decisions - cloud system for testing and time efficiency
     - Retrieval and ranking approach - evalation_results as a json format with a LLM as a judge
     - How context is prepared and fed to the LLM - as a chat completion with system and user structures
   - **Results and Analysis:**
     - Evaluation results - the system it yet to be implemented, not performing well
     - Analysis of strengths and weaknesses - 
     
     - Strenghts - chunking strategy, cloud vector database

     - Weaknesses - low accurary, chunking too large, not correlated with any other information within text as metadata such as names - places or people

     - Potential improvements to enhance performance and make this solution production-ready

      - 1. Better Chunking Strategy
      
      Implement semantic chunking instead of fixed-size chunks.

      Split text based on logical units (paragraphs, headings).

      - 2. Smarter Retrieval

      Hybrid Retrieval: Combine vector similarity search with keyword-based (BM25) search.

      Reranking: Apply a cross-encoder LLM to rerank top-k retrieved documents.

      Metadata Filtering: Filter results by metadata (e.g., date, author, tags).

      - 3. Try out different Embedding Models

      - 4. Query Reformulation and Expansion

      Example:

      Input: "Tea in Veridia"

      Expanded: "Types of tea, tea culture, and ceremonies in Veridia"

      - 5. Caching and Memory

      - 6. Vector Database Optimizations