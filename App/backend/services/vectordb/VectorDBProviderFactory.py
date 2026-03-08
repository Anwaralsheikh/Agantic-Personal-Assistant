from .providers import QdrantDBProvider, PineconeDBProvider, ChromaDBProvider
from .VectorDBEnums import VectorDBEnums
from controllers.BaseController import BaseController

class VectorDBProviderFactory:
    def __init__(self, config):
        self.config = config
        self.base_controller = BaseController()

    def create(self, provider: str):
        # 1. Qdrant
        if provider == VectorDBEnums.QDRANT.value:
            db_path = self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_PATH)
            return QdrantDBProvider(
                db_path=db_path,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
            )
            
        # 2. ChromaDB
        if provider == VectorDBEnums.CHROMA.value:
            # يفضل استخدام get_database_path أيضاً لضمان إنشاء المجلد محلياً
            db_path = self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_PATH)
            return ChromaDBProvider(
                db_path=db_path,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD
            )
        
        # 3. Pinecone (تعديل المستدعى هنا)
        elif provider == VectorDBEnums.PINECONE.value:
            return PineconeDBProvider(
                api_key=self.config.PINECONE_API_KEY, # تأكد من وجوده في السيتينغ
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD
            )
            
        return None