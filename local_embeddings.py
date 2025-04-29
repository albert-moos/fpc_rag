import requests
from langchain_core.embeddings import Embeddings
from typing import List

class LocalEmbeddings(Embeddings):
    """
    Custom Embeddings class
   POST request with a JSON
    {'texts': ['text1', 'text2', ...]} and returns a JSON body like
    {'embeddings': [[e1_1, e1_2, ...], [e2_1, e2_2, ...], ...]}.
    """
    def __init__(self, service_url: str):
        self.service_url = service_url

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        try:
            print(f"Sending {len(texts)} texts to local embedding service...")
            response = requests.post(self.service_url, json={'texts': texts}, timeout=30) # Added timeout
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            embeddings = response.json().get('embeddings')

            if not embeddings or len(embeddings) != len(texts) or not all(isinstance(e, list) and all(isinstance(val, (int, float)) for val in e) for e in embeddings):
                # check for the structure of embeddings
                raise ValueError(f"Invalid response format or content from embedding service. Expected list of lists of floats, got: {embeddings}")

            print(f"Received {len(embeddings)} embeddings.")
            return embeddings
        except requests.exceptions.Timeout:
             print(f"Timeout error from embedding service at {self.service_url}")
             raise
        except requests.exceptions.RequestException as e:
            print(f"Error embedding documents: {e}")
            raise e
        except ValueError as ve:
            print(f"Data validation error after receiving response: {ve}")
            raise ve


    def _embed_query(self, text: str) -> List[float]:
        """Embed a query text."""
        if not text:
             raise ValueError("embed query text is empty.")
        embeddings = self._embed_documents([text])
        if embeddings and len(embeddings) == 1:
            return embeddings[0]
        else:
            # Handle error if embedding failed
            raise ValueError("Could not embed query or received unexpected number of embeddings.")