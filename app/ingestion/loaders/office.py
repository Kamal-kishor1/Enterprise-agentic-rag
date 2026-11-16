import logfire


def parse_office(file_path: str):
    """
    Parse DOCX and PPTX files using their native Python libraries.
    """

    with logfire.span("Office Document Parsing", filename=file_path):
        try:
            if file_path.lower().endswith(".docx"):
                from docx import Document

                doc = Document(file_path)

                texts = [p.text for p in doc.paragraphs if p.text.strip()]

                text = "\n".join(texts)

                logfire.info(
                    f"python-docx parsed {len(text)} characters from {file_path}"
                )

                return text

            elif file_path.lower().endswith(".pptx"):
                from pptx import Presentation

                prs = Presentation(file_path)

                texts = []

                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text.strip():
                            texts.append(shape.text)

                text = "\n".join(texts)

                logfire.info(
                    f"python-pptx parsed {len(text)} characters from {file_path}"
                )

                return text

            else:
                logfire.warning(f"Unsupported Office file: {file_path}")
                return ""

        except Exception as e:
            logfire.error(f"Office Parse Failed for {file_path}: {e}")
            raise
