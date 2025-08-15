import re
from typing import List, Dict

def simple_chunk_text(text: str, max_chars: int = 900, overlap: int = 100) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]

def parse_docs(raw_text: str) -> List[Dict]:
    blocks = [b.strip() for b in raw_text.split("\n\n") if b.strip()]
    docs = []
    for b in blocks:
        title = None
        content = None
        for line in b.splitlines():
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip()
            elif line.lower().startswith("content:"):
                content = line.split(":", 1)[1].strip()
        if title and content:
            docs.append({"title": title, "content": content})
    return docs
