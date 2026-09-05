
# ============================================================
# Generic Chunking Module for Project LEDGER
# Based on the official doc-processor-api contract
# ============================================================


# ============================================================
# 1. Section Mapping
# ============================================================

def build_section_map(doc):
    """
    Build a mapping:

        section_id -> full section path

    Example:
        ["Annual Report 2019", "12. Income Taxes"]
        ->
        "Annual Report 2019 > 12. Income Taxes"
    """

    section_map = {}

    for section in doc.get("sections", []):

        section_id = section.get("section_id")

        if not section_id:
            continue

        path = section.get("path", [])

        if isinstance(path, list):
            section_map[section_id] = " > ".join(
                str(item) for item in path
            )
        else:
            section_map[section_id] = str(path)

    return section_map


# ============================================================
# 2. Normalize OCR Blocks
# ============================================================

def normalize_blocks(doc):
    """
    Convert the official doc-processor output into a common
    block representation used by Retrieval.

    This function is intentionally generic:
    - works with any number of pages
    - works with any number of blocks
    - preserves document/page/section/provenance metadata
    """

    if "document_id" not in doc:
        raise ValueError("Missing required field: document_id")

    section_map = build_section_map(doc)

    document_title = doc.get(
        "document_title",
        doc.get("filename", "")
    )

    normalized = []

    for page in doc.get("pages", []):

        page_number = page.get("page_number")

        for block in page.get("blocks", []):

            section_id = block.get("section_id")

            section = section_map.get(
                section_id,
                block.get("section", "")
            )

            content = block.get(
                "text",
                block.get("content", "")
            )

            content_type = block.get(
                "type",
                block.get("content_type")
            )

            bounding_box = block.get(
                "bbox",
                block.get("bounding_box")
            )

            record = {
                "document_id": doc["document_id"],
                "document_title": document_title,
                "page_number": block.get(
                    "page_number",
                    page_number
                ),
                "section": section,
                "section_id": section_id,
                "content": content or "",
                "content_type": content_type,
                "bounding_box": bounding_box,
                "table_id": block.get("table_id"),
                "block_id": block.get("block_id"),
                "order_index": block.get("order_index")
            }

            normalized.append(record)

    return normalized


# ============================================================
# 3. Section-Aware Chunking
# ============================================================

def section_aware_chunking(
    normalized_blocks,
    chunk_size=1000
):
    """
    Create text chunks while respecting section boundaries.

    Tables are excluded because they are indexed separately
    through table-aware chunking.

    The function works across the entire document/corpus.
    """

    chunks = []

    current_section = None
    current_text = []
    current_blocks = []

    def save_chunk():

        if not current_text:
            return

        content = " ".join(
            text.strip()
            for text in current_text
            if text.strip()
        )

        if not content:
            return

        pages = sorted({
            block.get("page_number")
            for block in current_blocks
            if block.get("page_number") is not None
        })

        bounding_boxes = [
            block.get("bounding_box")
            for block in current_blocks
            if block.get("bounding_box") is not None
        ]

        first_block = current_blocks[0]

        chunk = {
            "chunk_id": f"chunk_{len(chunks):06d}",
            "document_id": first_block["document_id"],
            "document_title": first_block["document_title"],

            # Internal representation keeps all pages
            # touched by the chunk.
            "page_number": pages,

            "section": current_section,
            "content": content,
            "content_type": "text",
            "table_id": None,
            "bounding_box": bounding_boxes
        }

        chunks.append(chunk)

    for block in normalized_blocks:

        content_type = block.get("content_type")

        # ----------------------------------------------------
        # Section headers
        # ----------------------------------------------------

        if content_type == "section_header":

            if current_section != block.get("section"):

                save_chunk()

                current_text = []
                current_blocks = []

                current_section = block.get("section")

            continue

        # ----------------------------------------------------
        # Tables are handled separately
        # ----------------------------------------------------

        if content_type == "table":
            continue

        # ----------------------------------------------------
        # Only textual content enters this strategy
        # ----------------------------------------------------

        if content_type != "text":
            continue

        text = block.get("content", "").strip()

        if not text:
            continue

        # ----------------------------------------------------
        # Section boundary
        # ----------------------------------------------------

        if current_section != block.get("section"):

            save_chunk()

            current_text = []
            current_blocks = []

            current_section = block.get("section")

        current_length = len(
            " ".join(current_text)
        )

        # ----------------------------------------------------
        # Chunk-size boundary
        # ----------------------------------------------------

        if (
            current_text
            and current_length + len(text) + 1 > chunk_size
        ):

            save_chunk()

            current_text = []
            current_blocks = []

        current_text.append(text)
        current_blocks.append(block)

    # Save final chunk
    save_chunk()

    return chunks


# ============================================================
# 4. Table -> Markdown
# ============================================================

def table_to_markdown(table):
    """
    Convert an OCR structured table into Markdown.

    Preserves:
    - rows
    - columns
    - cell text
    - headers
    - financial values as printed text
    """

    n_rows = table.get("n_rows", 0)
    n_cols = table.get("n_cols", 0)

    if n_rows <= 0 or n_cols <= 0:
        return ""

    matrix = [
        ["" for _ in range(n_cols)]
        for _ in range(n_rows)
    ]

    for cell in table.get("cells", []):

        row = cell.get("row")
        col = cell.get("col")

        if row is None or col is None:
            continue

        if not (0 <= row < n_rows):
            continue

        if not (0 <= col < n_cols):
            continue

        text = cell.get("text", "")

        if text is None:
            text = ""

        matrix[row][col] = str(text).strip()

    # Remove completely empty rows
    matrix = [
        row
        for row in matrix
        if any(cell.strip() for cell in row)
    ]

    if not matrix:
        return ""

    header = matrix[0]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(
            ["---"] * len(header)
        ) + " |"
    ]

    for row in matrix[1:]:

        # Make sure every row has the same number
        # of columns as the header.
        row = row[:len(header)]

        if len(row) < len(header):
            row += [""] * (
                len(header) - len(row)
            )

        lines.append(
            "| " + " | ".join(row) + " |"
        )

    return "\n".join(lines)


# ============================================================
# 5. Table-Aware Chunking
# ============================================================

def table_aware_chunking(doc):
    """
    Create one retrieval chunk per structured table.

    Tables are kept separate from text chunks because
    financial questions frequently depend on exact:
    - rows
    - columns
    - years
    - values
    - units
    """

    chunks = []

    section_map = build_section_map(doc)

    document_title = doc.get(
        "document_title",
        doc.get("filename", "")
    )

    for table in doc.get("tables", []):

        content = table_to_markdown(table)

        if not content.strip():
            continue

        section = section_map.get(
            table.get("section_id"),
            table.get("section", "")
        )

        chunk = {
            "chunk_id": f"table_chunk_{len(chunks):06d}",
            "document_id": doc["document_id"],
            "document_title": document_title,
            "page_number": table.get("page_number"),
            "section": section,
            "content": content,
            "content_type": "table",
            "table_id": table.get("table_id"),
            "bounding_box": table.get(
                "bbox",
                table.get("bounding_box")
            )
        }

        chunks.append(chunk)

    return chunks


# ============================================================
# 6. Parent Sections
# ============================================================

def build_parent_sections(doc):
    """
    Build logical parent sections from the OCR section tree.

    Parents are used for the parent-child retrieval experiment.
    """

    parents = []

    document_title = doc.get(
        "document_title",
        doc.get("filename", "")
    )

    for section in doc.get("sections", []):

        path = section.get("path", [])

        if isinstance(path, list):
            section_path = " > ".join(
                str(item) for item in path
            )
        else:
            section_path = str(path)

        parent = {
            "parent_id": f"parent_{len(parents):06d}",
            "document_id": doc["document_id"],
            "document_title": document_title,
            "section": section_path,
            "section_id": section.get("section_id"),
            "page_start": section.get("page_start"),
            "page_end": section.get("page_end")
        }

        parents.append(parent)

    return parents


# ============================================================
# 7. Parent-Child Chunking
# ============================================================

def parent_child_chunking(
    doc,
    chunk_size=500
):
    """
    Create smaller child chunks inside logical parent sections.

    Retrieval can operate on children while parent metadata
    provides broader section context.
    """

    parents = build_parent_sections(doc)

    children = []

    # --------------------------------------------------------
    # Helper: retrieve blocks belonging to a section
    # --------------------------------------------------------

    def get_section_blocks(section_id):

        section_blocks = []

        for page in doc.get("pages", []):

            for block in page.get("blocks", []):

                if block.get("section_id") != section_id:
                    continue

                if block.get("type") == "section_header":
                    continue

                if block.get("type") == "table":
                    continue

                text = (
                    block.get("text")
                    or block.get("content")
                    or ""
                ).strip()

                if text:
                    section_blocks.append(block)

        return section_blocks

    # --------------------------------------------------------
    # Process every parent section
    # --------------------------------------------------------

    for parent in parents:

        section_blocks = get_section_blocks(
            parent["section_id"]
        )

        current_blocks = []
        current_length = 0

        def create_child(blocks):

            if not blocks:
                return

            content_parts = []

            for block in blocks:

                text = (
                    block.get("text")
                    or block.get("content")
                    or ""
                ).strip()

                if text:
                    content_parts.append(text)

            content = " ".join(content_parts)

            if not content:
                return

            pages = sorted({
                block.get("page_number")
                for block in blocks
                if block.get("page_number") is not None
            })

            bounding_boxes = [
                block.get(
                    "bbox",
                    block.get("bounding_box")
                )
                for block in blocks
                if block.get(
                    "bbox",
                    block.get("bounding_box")
                ) is not None
            ]

            child = {
                "chunk_id": f"child_{len(children):06d}",
                "parent_id": parent["parent_id"],
                "document_id": parent["document_id"],
                "document_title": parent["document_title"],
                "page_number": pages,
                "section": parent["section"],
                "content": content,
                "content_type": "text",
                "table_id": None,
                "bounding_box": bounding_boxes
            }

            children.append(child)

        # ----------------------------------------------------
        # Build children according to chunk size
        # ----------------------------------------------------

        for block in section_blocks:

            text = (
                block.get("text")
                or block.get("content")
                or ""
            ).strip()

            if not text:
                continue

            if (
                current_blocks
                and current_length + len(text) + 1 > chunk_size
            ):

                create_child(current_blocks)

                current_blocks = []
                current_length = 0

            current_blocks.append(block)
            current_length += len(text) + 1

        create_child(current_blocks)

    return parents, children
