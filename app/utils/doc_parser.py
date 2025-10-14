from pdfminer.high_level import extract_text
from docx import Document

def extract_text_from_pdf(path: str) -> str:
    return extract_text(path)

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    texts = []
    for para in doc.paragraphs:
        texts.append(para.text)
    return "\n".join(texts)

def extract_text_from_file(path: str, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(path)
    elif filename.lower().endswith(".docx"):
        return extract_text_from_docx(path)
    else:
        raise ValueError("Unsupported file format")

