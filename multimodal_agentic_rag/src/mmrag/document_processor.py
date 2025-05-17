import io
from PyPDF2 import PdfReader
from unstructured.partition.pdf import partition_pdf
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

class MultimodalDocumentParser:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def process_pdf(self, file_path: str) -> List[DocumentChunk]:
        elements = partition_pdf(
            file_path,
            strategy="hi_res",
            infer_table_structure=True,
            include_page_blocks=True
        )
        
        chunks = []
        current_page = 0
        current_chunk = {"text": "", "images": []}
        
        for elem in elements:
            if elem.metadata.page_number != current_page:
                chunks.append(DocumentChunk(
                    text=current_chunk["text"],
                    images=current_chunk["images"],
                    metadata={"page": current_page},
                    chunk_id=f"page_{current_page}"
                ))
                current_page = elem.metadata.page_number
                current_chunk = {"text": "", "images": []}
            
            if elem.category == "Image":
                img = self._process_image(elem)
                current_chunk["images"].append(img)
            else:
                current_chunk["text"] += f"{elem.text}\n"
        
        return chunks

    def _process_image(self, elem):
        image = elem.metadata.image
        inputs = self.processor(image, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_length=50)
        elem.metadata.description = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return image
