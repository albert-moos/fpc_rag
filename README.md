Markdown

# 💬 FastAPI RAG-Based QA System (Azure OpenAI + Weaviate + Cohere)

This repository provides a sample implementation of a Retrieval-Augmented Generation (RAG) pipeline using FastAPI, Azure OpenAI (GPT-4), Weaviate, Cohere Rerank, and a local embedding service.

⚠️ This code is a sanitized sample developed during a contract project and complies with NDA constraints — no proprietary data is included.

## 🔧 Features

* RAG Pipeline using LangChain.
* Azure OpenAI GPT-4 for answer generation.
* Weaviate vector store for semantic search.
* Cohere Rerank for enhanced document relevance.
* Local embedding service integration.
* FastAPI backend with a single `/query` endpoint.

## 🚀 Quick Start

1.  **Clone the Repository**

    ```bash
    git clone [https://github.com/albert-moos/fpc_rag.git](https://github.com/albert-moos/fpc_rag.git)
    cd fpc_rag
    ```

2.  **Install Dependencies**

    It's recommended to use a virtual environment:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up Environment Variables**

    Create a `.env` file in the root directory with the following keys:

    ```env
    AZURE_OPENAI_API_KEY=your_azure_openai_key
    AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
    AZURE_OPENAI_API_VERSION=2023-05-15 # or your version
    AZURE_OPENAI_DEPLOYMENT_NAME=your_chat_model_name

    COHERE_API_KEY=your_cohere_api_key

    WEAVIATE_URL=http://x.x.x.x:22041
    WEAVIATE_API_KEY=your_weaviate_api_key # Can be empty if no key required
    WEAVIATE_INDEX_NAME=fpc_doc_full_doc # Match your Weaviate index name

    LOCAL_EMBEDDING_SERVICE_URL=http://x.x.x.x:22042
    ```

4.  **Run the Server**

    ```bash
    uvicorn main:app --reload
    ```

    The FastAPI server will be available at `http://127.0.0.1:8000`.

    You can test the API via:

    * Swagger UI: `http://127.0.0.1:8000/docs`
    * cURL or Postman via the `/query` POST endpoint.

## 📥 Sample Request

```json
POST /query
Content-Type: application/json

{
  "query": "What is the compliance process?"
}
```

## 🧱 Architecture

* Embedding: Local service via LocalEmbeddings.
* Storage: Vector index using Weaviate.
* Retrieval: Top-k document retrieval + reranking (via Cohere).
* LLM: Azure OpenAI GPT-4 (or GPT-3.5 as configured).
* API Layer: FastAPI.

## 🛠 Local Embedding Service

Ensure your embedding service is running at the specified URL (`LOCAL_EMBEDDING_SERVICE_URL`). It must expose a POST endpoint that accepts text payloads (likely a list of strings) and returns corresponding vector embeddings, in a format expected by the `LocalEmbeddings` class used in `main.py`.