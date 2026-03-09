from fastapi import FastAPI, APIRouter, Depends, UploadFile, status
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController
import aiofiles
from models import ResponseSignal
import logging
from .schemes.data import ProcessRequest,ChatRequest
from services.llm.LLMProviderFactory import LLMProviderFactory
from services.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from controllers.RagController import RagController

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    
)

@data_router.post("/upload/{project_id}")
async def upload_data(project_id: str, file: UploadFile,
                      app_settings: Settings = Depends(get_settings)):
        
    
    # validate the file properties
    data_controller = DataController()

    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)
    

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": result_signal
            }
        )
    # 

    project_dir_path = ProjectController().get_project_path(project_id=project_id)
    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id
    )

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:

        logger.error(f"Error while uploading file: {e}")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value
            }
        )

    return JSONResponse(
            content={
                "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
                "file_id": file_id
            }
        )
@data_router.post("/process/{project_id}")
async def process_endpoint(project_id: str, process_request: ProcessRequest):

    file_id = process_request.file_id
    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size

    process_controller = ProcessController(project_id=project_id)

    file_content = process_controller.get_file_content(file_id=file_id)

    file_chunks = process_controller.process_file_content(
        file_content=file_content,
        file_id=file_id,
        chunk_size=chunk_size,
        overlap_size=overlap_size
    )

    if file_chunks is None or len(file_chunks) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.PROCESSING_FAILED.value
            }
        )

    return file_chunks

@data_router.post("/index/{project_id}")
async def index_endpoint(project_id: str, process_request: ProcessRequest,
                         app_settings: Settings = Depends(get_settings)):

    llm_factory = LLMProviderFactory(app_settings)
    vdb_factory = VectorDBProviderFactory(app_settings)

    embed_provider = llm_factory.create(app_settings.EMBEDDING_BACKEND)
    vdb_provider = vdb_factory.create(app_settings.VECTOR_DB_BACKEND)

    embed_provider.set_embedding_model(
        model_id=app_settings.EMBEDDING_MODEL_ID,
        embedding_size=app_settings.EMBEDDING_MODEL_SIZE
    )
    vdb_provider.connect()

    rag_ctrl = RagController(
        vectordb_client=vdb_provider,
        generation_client=None,  
        embedding_client=embed_provider
    )


    is_indexed = rag_ctrl.index_project_file(
        project_id=project_id,
        file_id=process_request.file_id,
        chunk_size=process_request.chunk_size,
        overlap_size=process_request.overlap_size,
    )

    if not is_indexed:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.INDEXING_FAILED.value}
        )

    return JSONResponse(
        content={"signal": ResponseSignal.INDEXING_SUCCESS.value}
    )
@data_router.post("/chat/{project_id}")
async def chat_endpoint(project_id: str, chat_request: ChatRequest,
                        app_settings: Settings = Depends(get_settings)):

    llm_factory = LLMProviderFactory(app_settings)
    vdb_factory = VectorDBProviderFactory(app_settings)

    gen_provider = llm_factory.create(app_settings.GENERATION_BACKEND)
    embed_provider = llm_factory.create(app_settings.EMBEDDING_BACKEND)
    vdb_provider = vdb_factory.create(app_settings.VECTOR_DB_BACKEND)

    embed_provider.set_embedding_model(
        model_id=app_settings.EMBEDDING_MODEL_ID,
        embedding_size=app_settings.EMBEDDING_MODEL_SIZE
    )
    gen_provider.set_generation_model(model_id=app_settings.GENERATION_MODEL_ID)
    vdb_provider.connect()

    rag_ctrl = RagController(
        vectordb_client=vdb_provider,
        generation_client=gen_provider,
        embedding_client=embed_provider
    )

    try:
        answer = rag_ctrl.get_agent_response(
            project_id=project_id,
            user_question=chat_request.question,
            agent_type=chat_request.agent_type
        )
        return JSONResponse(content={
            "signal": ResponseSignal.CHAT_SUCCESS.value,
            "answer": answer
        })
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.CHAT_FAILED.value, "error": str(e)}
        )