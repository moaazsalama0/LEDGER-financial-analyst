import re


MAX_WORDS = 250
OVERLAP_WORDS = 50


def clean_text(text: str) -> str:
    """
    Normalize whitespace without changing the meaning.
    """
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(
    text: str,
    max_words: int = MAX_WORDS,
    overlap: int = OVERLAP_WORDS,
):
    """
    Split long text into overlapping word-based chunks.
    """

    text = clean_text(text)

    if not text:
        return []

    words = text.split()

    if len(words) <= max_words:
        return [text]

    chunks = []

    start = 0

    while start < len(words):

        end = start + max_words

        chunk_words = words[start:end]

        if chunk_words:
            chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_table_content(table: dict) -> str:
    """
    Convert OCR table data into searchable text.

    Prefer the complete markdown representation when available,
    because it preserves column headers, row values, and table
    structure.

    Falls back to column_headers, rows, and raw text when
    markdown is not available.
    """

    title = clean_text(table.get("title", ""))
    markdown = clean_text(table.get("markdown", ""))

    # Prefer the complete OCR markdown representation.
    if markdown:
        parts = []

        if title:
            parts.append(title)

        parts.append(markdown)

        return "\n".join(parts)

    # Fallback for tables without markdown.
    parts = []

    if title:
        parts.append(title)

    headers = table.get("column_headers")

    if headers:
        parts.append(
            "Columns: " + " | ".join(
                clean_text(str(header))
                for header in headers
            )
        )

    rows = table.get("rows")

    if rows:
        for row in rows:

            if isinstance(row, dict):
                parts.append(
                    " | ".join(
                        f"{clean_text(str(key))}: "
                        f"{clean_text(str(value))}"
                        for key, value in row.items()
                    )
                )

            elif isinstance(row, list):
                parts.append(
                    " | ".join(
                        clean_text(str(cell))
                        for cell in row
                    )
                )

            else:
                parts.append(
                    clean_text(str(row))
                )

    # Some OCR contracts may already provide table text.
    raw_text = clean_text(table.get("text", ""))

    if raw_text:
        parts.append(raw_text)

    return "\n".join(
        part
        for part in parts
        if part
    )


def build_section_chunks(document: dict):
    """
    Section-aware text chunks.

    Section headers are metadata and are not indexed as
    independent content.
    """

    chunks = []

    document_id = document["document_id"]
    document_title = document.get("document_title")

    sections = {
        section.get("section_id"): section
        for section in document.get("sections", [])
        if section.get("section_id") is not None
    }

    for page in document.get("pages", []):

        page_number = page.get("page_number")

        blocks = sorted(
            page.get("blocks", []),
            key=lambda b: (
                b.get("order_index")
                if b.get("order_index") is not None
                else 10**9
            )
        )

        for block in blocks:

            block_type = block.get("type")

            if block_type == "table":
                continue

            if block_type == "section_header":
                continue

            text = clean_text(
                block.get("text", "")
            )

            if not text:
                continue

            section_id = block.get("section_id")
            section = sections.get(section_id, {})

            section_title = section.get("title")
            section_path = section.get("path", [])

            text_parts = split_text(text)

            for chunk_idx, chunk_text in enumerate(text_parts):

                block_id = block.get("block_id")

                chunk_id = (
                    f"{document_id}_"
                    f"p{page_number}_"
                    f"b{block_id}_"
                    f"c{chunk_idx}"
                )

                chunks.append({
                    "chunk_id": chunk_id,

                    # Contract / source metadata
                    "document_id": document_id,
                    "document_title": document_title,
                    "page_number": page_number,

                    "section": section_title,
                    "section_path": section_path,

                    "content": chunk_text,
                    "content_type": "text",

                    "bounding_box": block.get("bbox"),
                    "table_id": None,

                    # Internal metadata
                    "source_block_ids": [block_id],
                    "chunk_type": "text",
                })

    return chunks


def build_table_chunks(document: dict):
    """
    Create one searchable chunk per table.

    Tables remain independent from text chunks so numerical
    and structured evidence is preserved.
    """

    chunks = []

    document_id = document["document_id"]
    document_title = document.get("document_title")

    sections = {
        section.get("section_id"): section
        for section in document.get("sections", [])
        if section.get("section_id") is not None
    }

    for table in document.get("tables", []):

        table_id = table.get("table_id")

        content = build_table_content(table)

        if not content:
            continue

        section_id = table.get("section_id")
        section = sections.get(section_id, {})

        section_title = section.get("title")
        section_path = section.get("path", [])

        page_number = table.get("page_number")

        chunk_id = (
            f"{document_id}_table_{table_id}"
        )

        chunks.append({
            "chunk_id": chunk_id,

            # Contract / source metadata
            "document_id": document_id,
            "document_title": document_title,
            "page_number": page_number,

            "section": section_title,
            "section_path": section_path,

            "content": content,
            "content_type": "table",

            "bounding_box": table.get("bbox"),
            "table_id": table_id,

            # Internal metadata
            "source_block_ids": [
                table.get("block_id")
            ],

            "chunk_type": "table",

            "n_rows": table.get("n_rows"),
            "n_cols": table.get("n_cols"),
            "column_headers": table.get("column_headers"),
            "row_headers": table.get("row_headers"),
            "units": table.get("units"),
        })

    return chunks


def build_retrieval_corpus(document: dict):
    """
    Final retrieval corpus:

        Section-aware text chunks
                    +
        Table-aware chunks
    """

    text_chunks = build_section_chunks(document)
    table_chunks = build_table_chunks(document)

    return text_chunks + table_chunks
