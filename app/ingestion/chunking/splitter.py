from typing import List
import logfire


def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """
    Robust chunker that splits on paragraphs (\n\n) or single lines (\n),
    ensuring chunks stay under chunk_size and no text is dropped.
    """
    with logfire.span(" Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        # Normalize carriage returns
        clean_text = text.replace("\r\n", "\n")

        # Split on double newlines if present, otherwise single newlines
        delimiter = "\n\n" if "\n\n" in clean_text else "\n"
        elements = [e.strip() for e in clean_text.split(delimiter) if e.strip()]

        chunks = []
        current_chunk = ""

        for el in elements:
            # If a single line/element exceeds chunk_size, split by words
            if len(el) > chunk_size:
                words = el.split(" ")
                temp_buf = ""
                for w in words:
                    if len(temp_buf) + len(w) + 1 < chunk_size:
                        temp_buf += (" " if temp_buf else "") + w
                    else:
                        if temp_buf:
                            chunks.append(temp_buf)
                        temp_buf = w
                if temp_buf:
                    chunks.append(temp_buf)
                continue

            if len(current_chunk) + len(el) + 2 < chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + el
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = el

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f" Generated {len(valid_chunks)} chunks")
        return valid_chunks
