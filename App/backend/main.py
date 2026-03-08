import os
import pydantic
if pydantic.__version__.startswith("2."):
    from pydantic.v1 import SecretStr
from helpers.config import get_settings
from services.llm.LLMProviderFactory import LLMProviderFactory
from services.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from controllers.RagController import RagController
import logging
import pdb



logging.basicConfig(level=logging.INFO)

def setup_langsmith(settings):
    if settings.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT or "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT or "rag-assistant"

        print("LangSmith is active")
        print(f"https://smith.langchain.com")
    else:
        print("LangSmith is not active")

def test_rag_system():
    
    settings = get_settings()
    setup_langsmith(settings)
    
    
    llm_factory = LLMProviderFactory(settings)
    vdb_factory = VectorDBProviderFactory(settings)
    
    gen_provider = llm_factory.create(settings.GENERATION_BACKEND)
    embed_provider = llm_factory.create(settings.EMBEDDING_BACKEND)
    vdb_provider = vdb_factory.create(settings.VECTOR_DB_BACKEND)

    embed_provider.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID, 
        embedding_size=settings.EMBEDDING_MODEL_SIZE
    )

    test_vector = embed_provider.embed_text("test")
    print(f"Embedding size: {len(test_vector)}")
    
    gen_provider.set_generation_model(model_id = settings.GENERATION_MODEL_ID)
    
    vdb_provider.connect()
    print(f"VectorDB connected: {vdb_provider}")
    print(f"Embed provider: {embed_provider}")
    
    rag_ctrl = RagController(
        vectordb_client=vdb_provider,
        generation_client=gen_provider,
        embedding_client=embed_provider
    )
    

    project_id = "test_project_002"
    
    
    file_id = "test.pdf" 
    
    print("--- (Indexing) ---")
    try:
        is_indexed = rag_ctrl.index_project_file(
            project_id=project_id, 
            file_id=file_id
        )
        if is_indexed:
            print("inedxing done successfuly")
        else:
            print("indexing failed")
            return
    except Exception as e:
        print(f"indexing error {e}")
        return

    print("\n--- (Agent Chat) ---")
    question = "what is AI Agent?" 
    
    try:
        answer = rag_ctrl.get_agent_response(
            project_id=project_id, 
            user_question=question
        )
        print(f"\n Agent Answer:\n{answer}")
    except Exception as e:
        print(f"error during Agent calling: {e}")

if __name__ == "__main__":
    test_rag_system()

