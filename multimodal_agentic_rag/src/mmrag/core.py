
from pydantic import BaseModel
from typing import List, Dict, Optional
from PIL import Image

class DocumentChunk(BaseModel):
    text: str
    images: List[Image.Image]
    metadata: Dict[str, str]
    chunk_id: str

class ImageAnalysisResult(BaseModel):
    content_type: str  # 'chart', 'diagram', 'photograph', 'unknown'
    relevance_score: float
    needs_human_review: bool = False

class RetrievedContext(BaseModel):
    text_context: List[str]
    image_context: List[ImageAnalysisResult]
    confidence: float

class VerifiedResponse(BaseModel):
    answer: str
    confidence: float
    uncertainty_reasons: List[str]
    citations: List[str]