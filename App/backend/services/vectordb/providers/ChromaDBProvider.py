import chromadb
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
import logging
from typing import List

class ChromaDBProvider(VectorDBInterface):

    def __init__(self, db_path: str, distance_method: str):
        self.db_path = db_path
        self.client = None
        
        # Chroma يستخدم مسميات مختلفة للمسافات
        # 'l2' للهندسي، 'cosine' للتشابه، 'ip' للضرب النقطي
        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_metric = "cosine"
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_metric = "ip" # Inner Product
        else:
            self.distance_metric = "l2"
            
        self.logger = logging.getLogger(__name__)

    def connect(self):
        # PersistentClient يضمن حفظ البيانات في المجلد المحدد
        self.client = chromadb.PersistentClient(path=self.db_path)

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        # Chroma يعطي خطأ إذا طلبت كولكشن غير موجود، لذا نتحقق من القائمة
        collections = self.client.list_collections()
        return any(c.name == collection_name for c in collections)
    
    def list_all_collections(self) -> List:
        return [c.name for c in self.client.list_collections()]
    
    def get_collection_info(self, collection_name: str) -> dict:
        return self.client.get_collection(name=collection_name).get()
    
    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            self.client.delete_collection(name=collection_name)
        
    def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False):
        if do_reset:
            self.delete_collection(collection_name)
        
        # Chroma لا يتطلب embedding_size صراحة عند الإنشاء، بل يحدده عند أول إضافة
        self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )
        return True
    
    def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, 
                         record_id: str = None):
        
        collection = self.client.get_collection(name=collection_name)
        record_id = record_id or "id_1"
        
        try:
            collection.add(
                ids=[record_id],
                embeddings=[vector],
                documents=[text],
                metadatas=[metadata or {}]
            )
            return True
        except Exception as e:
            self.logger.error(f"Error in Chroma insert_one: {e}")
            return False
    
    def insert_many(self, collection_name: str, texts: list, vectors: list, metadata: list = None):
        collection = self.client.get_collection(name=collection_name)
        
        # Chroma يتوقع قائمة من الـ IDs
        ids = [f"id_{i}" for i in range(len(texts))]
        
        try:
            collection.add(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadata if metadata else [{}] * len(texts)
            )
            return True
        except Exception as e:
            self.logger.error(f"Error in Chroma insert_many: {e}")
            return False
        
    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):
        collection = self.client.get_collection(name=collection_name)
        return collection.query(
            query_embeddings=[vector],
            n_results=limit
        )

    def get_langchain_retriever(self, collection_name: str, embeddings_model, search_kwargs: dict = None) -> VectorStoreRetriever:
        # الجسر مع LangChain باستخدام مكتبة langchain-chroma
        vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=embeddings_model
        )
        return vector_store.as_retriever(search_kwargs=search_kwargs or {"k": 5})