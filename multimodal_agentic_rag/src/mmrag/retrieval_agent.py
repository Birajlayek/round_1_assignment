# mma_rag/retrieval_agent.py
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np

class ContextRetrievalAgent:
    def __init__(self):
        self.text_embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.vector_store = FAISS(embedding_function=self.text_embedder)
        
    def index_documents(self, chunks: List[DocumentChunk]):
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        self.vector_store.add_texts(texts, metadatas=metadatas)
        
    def retrieve_context(self, query: str, k: int = 5) -> RetrievedContext:
        docs = self.vector_store.similarity_search(query, k=k)
        return RetrievedContext(
            text_context=[doc.page_content for doc in docs],
            image_context=[],
            confidence=self._calculate_confidence(docs)
        )
    
    def _calculate_confidence(self, docs):
        return min(1.0, len(docs) / 5)
