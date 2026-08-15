from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF
from docx import Document


DOCUMENTS_DIR = Path(__file__).resolve().parents[2] / "documents"


def load_pdf(file_path: Path) -> List[Dict]:
    """
    Load text from a PDF document page by page.
    """

    documents = []

    pdf = fitz.open(file_path)

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()

        if not text:
            continue

        documents.append(
            {
                "text": text,
                "metadata": {
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "file_type": "pdf",
                    "page": page_number,
                },
            }
        )

    pdf.close()

    return documents


def load_docx(file_path: Path) -> List[Dict]:
    """
    Load text from a DOCX document.
    """

    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    text = "\n".join(paragraphs)

    if not text:
        return []

    return [
        {
            "text": text,
            "metadata": {
                "source": file_path.name,
                "file_path": str(file_path),
                "file_type": "docx",
            },
        }
    ]


def load_txt(file_path: Path) -> List[Dict]:
    """
    Load text from a TXT document.
    """

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).strip()

    if not text:
        return []

    return [
        {
            "text": text,
            "metadata": {
                "source": file_path.name,
                "file_path": str(file_path),
                "file_type": "txt",
            },
        }
    ]


def load_document(file_path: Path) -> List[Dict]:
    """
    Load a single document based on its extension.
    """

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension == ".docx":
        return load_docx(file_path)

    if extension == ".txt":
        return load_txt(file_path)

    return []


def load_all_documents() -> List[Dict]:
    """
    Load all supported documents from the documents folder.
    """

    all_documents = []

    if not DOCUMENTS_DIR.exists():
        print(f"Documents directory not found: {DOCUMENTS_DIR}")
        return []

    supported_extensions = {
        ".pdf",
        ".docx",
        ".txt",
    }

    files = [
        file
        for file in DOCUMENTS_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in supported_extensions
    ]

    for file_path in files:
        print(f"Loading: {file_path.name}")

        documents = load_document(file_path)

        all_documents.extend(documents)

    return all_documents