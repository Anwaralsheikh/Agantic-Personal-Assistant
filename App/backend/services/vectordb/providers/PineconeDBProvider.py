from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
from langchain_core.vectorstores import VectorStoreRetriever
import logging
from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


class PineconeDBProvider(VectorDBInterface):

    def __init__(self, api_key: str, environment: str, distance_method: str):
        
        self.client = None
        self.api_Key = api_key
        self.index = None
        self.distance_method = distance_method # 'cosine', 'dotproduct', 'euclidean'
        
        

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_metric = "cosine"
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_metric = "dotproduct"
        else:
            self.distance_metric = "euclidean"
        
        self.logger = logging.getLogger(__name__)

    def connect(self):
        self.client = Pinecone(api_key=self.api_key)

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        # في بينكون تسمى Indexes
        existing_indexes = [idx.name for idx in self.client.list_indexes()]
        return collection_name in existing_indexes
    
    def list_all_collections(self) -> List:
        return [idx.name for idx in self.client.list_indexes()]
    
    def get_collection_info(self, collection_name: str) -> dict:
        return self.client.describe_index(collection_name)
    
    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            return self.client.delete_index(collection_name)
        
    def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False):
        if do_reset and self.is_collection_existed(collection_name):
            self.client.delete_index(collection_name)
        
        if not self.is_collection_existed(collection_name):
            self.client.create_index(
                name=collection_name,
                dimension=embedding_size,
                metric=self.distance_metric,
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            return True
        return False
    
    def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, 
                         record_id: str = None):
        
        if not self.is_collection_existed(collection_name):
            return False
        
        index = self.client.Index(collection_name)
        record_id = record_id or "id_1"
        
        try:
            index.upsert(vectors=[{
                "id": record_id,
                "values": vector,
                "metadata": {"text": text, **(metadata or {})}
            }])
            return True
        except Exception as e:
            self.logger.error(f"Error in Pinecone insert_one: {e}")
            return False
    
    def insert_many(self, collection_name: str, texts: list, vectors: list, metadata: list = None):
        index = self.client.Index(collection_name)
        records = []
        
        for i in range(len(texts)):
            # تجهيز البيانات بصيغة Pinecone (id, values, metadata)
            records.append({
                "id": f"vec_{i}", # أو استخدم UUID
                "values": vectors[i],
                "metadata": {
                    "text": texts[i], 
                    **(metadata[i] if metadata[i] else {})
                }
            })
            
        try:
            # Pinecone يستخدم upsert وليس upload_records
            index.upsert(vectors=records)
            return True
        except Exception as e:
            self.logger.error(f"Error in Pinecone upsert: {e}")
            return False
        
    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):
        index = self.client.Index(collection_name)
        return index.query(
            vector=vector,
            top_k=limit,
            include_metadata=True
        )
    def get_langchain_retriever(self, collection_name: str, embeddings_model, search_kwargs: dict = None):
        vector_store = PineconeVectorStore(
            index_name=collection_name, 
            embedding=embeddings_model, 
            pinecone_api_key=self.api_key
        )
        return vector_store.as_retriever(search_kwargs=search_kwargs or {"k": 5})