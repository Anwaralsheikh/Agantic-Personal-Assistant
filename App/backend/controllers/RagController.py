from .BaseController import BaseController
from .ProcessController import ProcessController
from Agents.react import RagAgent
from langchain_core.tools.retriever import create_retriever_tool
import logging

class RagController(BaseController):
    
    def __init__(self, vectordb_client, generation_client, embedding_client):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.logger = logging.getLogger(__name__)

    def get_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()

    # --- (Indexing) ---
    
    def index_project_file(self, project_id: str, file_id: str, chunk_size: int = 500, overlap_size: int = 50):
        
        
        processor = ProcessController(project_id=project_id)
        file_content = processor.get_file_content(file_id=file_id)
        
        if not file_content:
            self.logger.error(f"Could not load file content for: {file_id}")
            return False

        chunks = processor.process_file_content(
            file_content=file_content,
            file_id=file_id,
            chunk_size=chunk_size,
            overlap_size=overlap_size
        )

    
        texts = [c.page_content for c in chunks]
        metadata = [c.metadata for c in chunks]
        vectors = [self.embedding_client.embed_text(text) for text in texts]

    
        collection_name = self.get_collection_name(project_id)
        
        self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=len(vectors[0]),
            do_reset=False 
        )

        return self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors
        )

    

    def get_agent_response(self, project_id: str, user_question: str):
        
        collection_name = self.get_collection_name(project_id)
        embeddings_model = self.embedding_client.get_langchain_embeddings()
        retriever = self.vectordb_client.get_langchain_retriever(
            collection_name=collection_name,
            embeddings_model=embeddings_model
        )

        search_tool = create_retriever_tool(
            retriever,
            "project_search_tool",
            
            """Search and retrieve relevant information from the project documents.
            Use this tool whenever the user asks any question about the document content.
            Always use this tool FIRST before answering any question."""
        )

        agent_logic = RagAgent(
            llm_model=self.generation_client.get_langchain_model(),
            tools=[search_tool]
        )
        
        executor = agent_logic.get_executor()

        response = executor.invoke({
            "input": user_question,
            "chat_history": [] 
        })

        return response["output"]