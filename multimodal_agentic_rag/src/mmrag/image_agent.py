# mma_rag/image_agent.py
from transformers import CLIPProcessor, CLIPModel
import torch

class ImageAnalysisAgent:
    CONTENT_TYPES = ["chart", "diagram", "photograph", "unknown"]
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        
    def analyze_image(self, image) -> ImageAnalysisResult:
        inputs = self.clip_processor(
            text=self.CONTENT_TYPES,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        
        content_type = self.CONTENT_TYPES[probs.argmax().item()]
        relevance = self._calculate_relevance(content_type)
        
        return ImageAnalysisResult(
            content_type=content_type,
            relevance_score=relevance,
            needs_human_review=relevance > 0.7
        )

    def _calculate_relevance(self, content_type):
        return 0.9 if content_type in ["chart", "diagram"] else 0.3
