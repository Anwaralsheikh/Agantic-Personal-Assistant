from enum import Enum

class VectorDBEnums(Enum):
    QDRANT = "QDRANT"
    PINECONE = "PINECONE"
    FAISS = "FAISS"
    CHROMA = "CHROMA"

class DistanceMethodEnums(Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"