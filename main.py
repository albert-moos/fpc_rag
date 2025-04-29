import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Weaviate
from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from local_embeddings import LocalEmbeddings # Import LocalEmbeddings from another file
import weaviate
from typing import List, Dict, Any

# Load environment stuff from a .env file
# AZURE_OPENAI_API_KEY="your_azure_openai_key"
# AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
# AZURE_OPENAI_API_VERSION="your_azure_openai_api_version" # like "2023-05-15"
# AZURE_OPENAI_DEPLOYMENT_NAME="your_azure_openai_deployment_name" # for chat model
# COHERE_API_KEY="your_cohere_api_key"
# WEAVIATE_URL="http://x.x.x.x:22041"
# WEAVIATE_API_KEY="your_weaviate_api_key"
# WEAVIATE_INDEX_NAME="main_rag_fpc"
# LOCAL_EMBEDDING_SERVICE_URL="http://x.x.x.x:22042"
load_dotenv()

# --- Test Config ---
# Azure OpenAI
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_openai_chat_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Cohere key
cohere_api_key = os.getenv("COHERE_API_KEY")

# Weaviate connection info
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY") # Can be None if not required
weaviate_index_name = os.getenv("WEAVIATE_INDEX_NAME")

# Local Embedding Service URL
local_embedding_service_url = os.getenv("LOCAL_EMBEDDING_SERVICE_URL")

# Quick check to see if we're missing anything crucial
required_vars = {
    "AZURE_OPENAI_API_KEY": azure_openai_api_key,
    "AZURE_OPENAI_ENDPOINT": azure_openai_endpoint,
    "AZURE_OPENAI_API_VERSION": azure_openai_api_version,
    "AZURE_OPENAI_DEPLOYMENT_NAME": azure_openai_chat_deployment_name,
    "COHERE_API_KEY": cohere_api_key,
    "WEAVIATE_URL": weaviate_url,
    "WEAVIATE_INDEX_NAME": weaviate_index_name,
    "LOCAL_EMBEDDING_SERVICE_URL": local_embedding_service_url
}

missing_vars = [var_name for var_name, value in required_vars.items() if not value or value.strip() == ""]
if missing_vars:
    print(f"Error: Looks like these environment variables are missing or empty: {', '.join(missing_vars)}")


# --- RAG Pipeline ---
# local embedding
embeddings = None
try:
    print(f"Trying to fire up the local embedding service at {local_embedding_service_url}...")
    embeddings = LocalEmbeddings(service_url=local_embedding_service_url)
    print("Local embedding found!")
except Exception as e:
    print(f"Couldn't initialize the local embedding service: {e}")


# initialize Weaviate and vector store
client = None
vectorstore = None
try:
    print(f"Connect to Weaviate at {weaviate_url}...")
    auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key) if weaviate_api_key else None
    client = weaviate.Client(
        url=weaviate_url,
        auth_client_secret=auth_config,
        connection_timeout_config=(5, 15) # Need to adjust longer in production (Ting_wei)
    )
    if not client.is_ready():
        raise ConnectionError("Weaviate client failed.")
    print("Weaviate client connected.")

    print(f"Setting up the Weaviate vector store using index (fpc_doc_full_doc)...")
    if embeddings is None:
         raise ValueError("Embedding service not initialized check the 22042 (POST)!")

    vectorstore = Weaviate(
        client=client,
        index_name="fpc_doc_full_doc",
        text_key="text",
        embedding=embeddings,
        by_text=False
    )
    print("Weaviate vector store initialized and ready.")
except Exception as e:
    print(f"Couldn't connect to Weaviate or set up the vector store: {e}")
    # This is probably a showstopper


# Get the retriever ready (first pass at finding docs)
retriever = None
if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20}) # Grab the top 20 docs (first phase)
    print(f"Retriever initialized to snag the top {retriever.search_kwargs['k']} documents.")
else:
    print("No vector store")


# Initialize the Reranker (second phase)
reranker = None
try:
    print("Getting Cohere Rerank ready...")
    reranker = CohereRerank(cohere_api_key=cohere_api_key)
    print("Cohere Rerank is set.")
except Exception as e:
    print(f"Failed to initialize Cohere Rerank. {e}")


# Combine the retriever and reranker
compression_retriever = None
if retriever and reranker:
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=retriever,
        document_compressor=reranker
    )
    print("Contextual Compression Retriever is combined.")
else:
    print("Can't build the compression retriever.")


# Initialize the Azure OpenAI Chat gpt4o
llm = None
try:
    print(f"Chat Model deployment '{azure_openai_chat_deployment_name}'...")
    llm = AzureChatOpenAI(
        azure_deployment=azure_openai_chat_deployment_name,
        openai_api_version=azure_openai_api_version,
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_api_key,
        temperature=0
    )
    print("Azure OpenAI Chat Model is set.")
except Exception as e:
    print(f"Couldn't initialize Azure OpenAI: {e}")


# RAG prompt
prompt = ChatPromptTemplate.from_template("""You are an expert assistant tasked with answering questions based strictly on the provided context documents.
Instructions:

Use only the information contained in the context to answer the question.
Do not use outside knowledge, assumptions, or general world facts.
Every part of your answer must be directly supported by the content of the documents.
If the answer is not clearly supported by the documents, respond exactly with:
"The provided documents do not contain this information."
If the documents provide incomplete or ambiguous information, clearly indicate this in your answer.
Do not paraphrase in a way that introduces new meaning; stay close to the original wording where possible.
Context:

{context}
Question:

{input}
Answer:""")
print("Chat prompt template is ready.")


# Build the Langchain chains
document_chain = None
retrieval_chain = None
if llm and compression_retriever:
    document_chain = create_stuff_documents_chain(llm, prompt)
    print("Stuff documents chain built.")
    retrieval_chain = create_retrieval_chain(compression_retriever, document_chain) # Retriever feeds the document chain
    print("Retrieval chain is complete.")
else:
    print("Can't build the chains.")


# --- FastAPI App Time! ---
app = FastAPI()

#(user's query)
class QueryRequest(BaseModel):
    query: str

# Define source documents
class SourceDocument(BaseModel):
    content: str # The actual text from the doc
    metadata: Dict[str, Any] = {} # Any extra info about the doc

# Define whole response
class RAGResponse(BaseModel):
    answer: str # The generated answer
    source_documents: List[SourceDocument] # A list of the docs used


# main endpoint for queries
@app.post("/query", response_model=RAGResponse) # POST request to /query, and it returns RAGResponse format
async def process_query(request: QueryRequest):
    # Check if the pipeline is ready
    if retrieval_chain is None:
        print("RAG pipeline wasn't initialized!")
        raise HTTPException(status_code=500, detail="RAG pipeline is not ready. Check server logs.")

    query = request.query
    print(f"\nquery received: '{query}'")

    try:
        # Run the RAG chain
        response = retrieval_chain.invoke({"input": query})

        # Get the answer
        answer = response.get("answer", "Couldn't generate an answer based on the documents.")
        # Get the source docs
        context_docs = response.get("context", [])

        # Format those source docs
        source_documents_list = []
        if context_docs:
            for doc in context_docs:
                source_documents_list.append(SourceDocument(content=doc.page_content, metadata=doc.metadata))

        print(f"Returning the answer and {len(source_documents_list)} source documents.")

        return RAGResponse(answer=answer, source_documents=source_documents_list)

    except Exception as e:
        print(f"\nSomething went wrong while processing the query: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {e}")