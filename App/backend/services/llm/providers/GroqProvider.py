from ..LLMInterface import LLMInterface
from ..LLMEnums import DocumentTypeEnum
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from groq import Groq
import logging

class GroqProvider(LLMInterface):

    def __init__(self, api_key: str,
                       default_input_max_characters: int=1000,
                       default_generation_max_output_tokens: int=1000,
                       default_generation_temperature: float=0.1):
        
        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = Groq(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        # ⚠️ Groq لا يدعم Embedding، نستخدم Cohere أو Gemini للـ Embedding
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.logger.warning("Groq does not support embeddings. Use Cohere or Gemini for embeddings.")

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list=[],
                      max_output_tokens: int=None, temperature: float=None):

        if not self.generation_model_id:
            self.logger.error("Generation model for Groq was not set")
            return None

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature or self.default_generation_temperature

        messages = chat_history + [
            {"role": "user", "content": self.process_text(prompt)}
        ]

        response = self.client.chat.completions.create(
            model=self.generation_model_id,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature,
        )

        if not response or not response.choices:
            self.logger.error("Error while generating text with Groq")
            return None

        return response.choices[0].message.content

    def embed_text(self, text: str, document_type: str=None):
        # Groq لا يدعم Embedding
        self.logger.error("Groq does not support embed_text. Use Cohere or Gemini instead.")
        return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }

    def get_langchain_model(self) -> BaseChatModel:
        return ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.generation_model_id or "llama-3.3-70b-versatile",
            temperature=self.default_generation_temperature,
            max_tokens=self.default_generation_max_output_tokens,
        )

    def get_langchain_embeddings(self) -> Embeddings:
        # Groq لا يدعم Embeddings
        self.logger.error("Groq does not support embeddings.")
        return None