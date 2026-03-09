from pydantic import BaseModel
from typing import Optional

class ProcessRequest(BaseModel):
    file_id: str
    chunk_size: Optional[int] = 100
    overlap_size: Optional[int] = 20
    do_reset: Optional[int] = 0

class ChatRequest(BaseModel):
    question: str
    agent_type: Optional[str] = "tool_calling"  # "react" أو "tool_calling"