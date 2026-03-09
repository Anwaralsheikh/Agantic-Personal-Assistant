from pydantic import BaseModel, Field, validator
from typing import Optional, Dict

    
class RetrievedDocument(BaseModel):
    text: str
    score: float
    metadata: Optional[Dict] = None