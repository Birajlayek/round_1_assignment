# mma_rag/orchestrator.py
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

class OrchestrationAgent:
    def __init__(self):
        self.document_parser = MultimodalDocumentParser()
        self.image_agent = ImageAnalysisAgent()
        self.retrieval_agent = ContextRetrievalAgent()
        self.verifier = HallucinationVerifier()
        self.llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-xxl",
            task="text2text-generation",
            device=0 if torch.cuda.is_available() else -1
        )
        
    def process_query(self, query: str, document_path: str) -> VerifiedResponse:
        chunks = self.document_parser.process_pdf(document_path)
        self.retrieval_agent.index_documents(chunks)
        
        context = self.retrieval_agent.retrieve_context(query)
        draft_response = self._generate_response(query, context)
        
        return self.verifier.verify_response(draft_response, context)
    
    def _generate_response(self, query: str, context: RetrievedContext) -> str:
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retrieval_agent.vector_store.as_retriever()
        )
        return qa_chain.run(query)
