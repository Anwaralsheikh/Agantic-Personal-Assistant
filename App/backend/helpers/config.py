from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str

    FILE_ALLOWED_TYPES: list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    MONGODB_URL: str
    MONGODB_DATABASE: str

    

    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str

    OPENAI_API_KEY: str = None
    OPENAI_API_URL: str = None
    COHERE_API_KEY: str = None
    GOOGLE_API_KEY: str = None
    GROQ_API_KEY: str = None

    PINECONE_API_KEY: str = None
    PINECONE_ENVIRONMENT: str = None
    PINECONE_INDEX_NAME: str = None

    LANGSMITH_TRACING: str= None
    LANGSMITH_ENDPOINT: str = None
    LANGSMITH_API_KEY: str = None
    LANGSMITH_PROJECT: str = None

    GENERATION_MODEL_ID: str = None
    EMBEDDING_MODEL_ID: str = None
    EMBEDDING_MODEL_SIZE: int = None
    INPUT_DEFAULT_MAX_CHARACTERS: int = None
    GENERATION_DEFAULT_MAX_TOKENS: int = None
    GENERATION_DEFAULT_TEMPERATURE: float = None

    VECTOR_DB_BACKEND : str
    VECTOR_DB_PATH : str
    VECTOR_DB_DISTANCE_METHOD: str = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",          # تجاهل أي متغيرات إضافية لا نحتاجها
        case_sensitive=False     # لا تهتم إذا كان المتغير كبيراً أو صغيراً
    )
    # class Config:
    #     env_file = ".env"

def get_settings():
    return Settings()