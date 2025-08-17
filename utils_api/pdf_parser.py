"""
PDF parser: reads PDF bytes and extracts text.
Input: bytes; Output: list[str] (lines). Used before chunking/embedding.
"""
from PyPDF2 import PdfReader
import io


class PDFParser:
    def parse_pdf_from_bytes(self, file_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text.strip()
        
def parse_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip().split("\n")  # returns list of lines
