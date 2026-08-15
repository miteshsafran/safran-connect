from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Employee question",
    )


class SourceResponse(BaseModel):
    id: int
    document: str
    page: Optional[int] = None
    chunk_id: Optional[int] = None
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]