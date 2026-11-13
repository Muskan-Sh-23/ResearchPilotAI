import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def clean_text(text):
    # Re-join words split by a hyphen at a line break: "classi-\nfication" -> "classification"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # PDFs break lines mid-sentence; turn single newlines into spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse repeated spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_text(text):
    chunks = _splitter.split_text(clean_text(text))
    # Drop tiny fragments (page numbers, stray headers)
    return [c for c in chunks if len(c.strip()) > 80]