"""Build small, valid PDFs in memory for tests.

Using synthesised PDFs rather than checked-in binaries keeps the test suite
deterministic and lets a test state the exact text and coordinates it expects
to read back, which is what makes the geometry assertions meaningful.

The generator writes a correct cross-reference table rather than relying on a
reader's error recovery, so these fixtures exercise the same code path a real
document does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: US Letter in PDF points.
PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0


@dataclass(frozen=True)
class TextItem:
    """One string drawn on a page.

    ``x`` and ``y`` are in PDF user space: origin bottom-left, y increasing
    upward. That is the PDF's own convention, deliberately kept here so tests
    exercise the service's flip into the contract's top-left convention.
    """

    text: str
    x: float
    y: float
    size: float = 11.0
    font: str = "F1"


@dataclass
class PageSpec:
    """One page of a synthesised document."""

    items: list[TextItem] = field(default_factory=list)
    width: float = PAGE_WIDTH
    height: float = PAGE_HEIGHT


def _escape(text: str) -> str:
    """Escape a string for a PDF literal-string operand."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(page: PageSpec) -> bytes:
    parts = ["BT"]
    for item in page.items:
        parts.append(f"/{item.font} {item.size} Tf")
        parts.append(f"1 0 0 1 {item.x:.2f} {item.y:.2f} Tm")
        parts.append(f"({_escape(item.text)}) Tj")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1")


def build_pdf(pages: list[PageSpec]) -> bytes:
    """Assemble a multi-page PDF using Helvetica, returning its bytes."""
    if not pages:
        raise ValueError("a PDF needs at least one page")

    objects: list[bytes] = []

    def add(body: bytes) -> int:
        """Append an object and return its 1-based object number."""
        objects.append(body)
        return len(objects)

    # Reserve 1 for the catalog and 2 for the page tree so the kids list can
    # be written once the page objects exist.
    objects.append(b"")  # 1: catalog, filled in below
    objects.append(b"")  # 2: page tree, filled in below

    font_num = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_nums: list[int] = []
    for page in pages:
        stream = _content_stream(page)
        contents_num = add(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
        page_num = add(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page.width:.2f} {page.height:.2f}] "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> "
            f"/Contents {contents_num} 0 R >>".encode()
        )
        page_nums.append(page_num)

    kids = " ".join(f"{n} 0 R" for n in page_nums)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_nums)} >>".encode()
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    count = len(objects) + 1
    out += f"xref\n0 {count}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    return bytes(out)


def simple_report_pdf() -> bytes:
    """A two-page stand-in for a financial report excerpt.

    Page 1 carries a title and a paragraph; page 2 carries a section heading
    and figures that exercise the finance parser, including a parenthesised
    negative and a dash.
    """
    page1 = PageSpec(
        items=[
            TextItem("Annual Report 2019", x=72.0, y=720.0, size=20.0),
            TextItem("Consolidated Statements of Income", x=72.0, y=680.0, size=14.0),
            TextItem(
                "Total revenue increased 7% in 2019, driven by subscription growth.",
                x=72.0,
                y=640.0,
            ),
            TextItem("Page 1", x=290.0, y=40.0, size=9.0),
        ]
    )
    page2 = PageSpec(
        items=[
            TextItem("12. Income Taxes", x=72.0, y=720.0, size=14.0),
            TextItem("(in millions)", x=72.0, y=700.0, size=9.0),
            TextItem("Current federal", x=72.0, y=670.0),
            TextItem("1,234", x=300.0, y=670.0),
            TextItem("(567)", x=400.0, y=670.0),
            TextItem("Deferred", x=72.0, y=650.0),
            TextItem("890", x=300.0, y=650.0),
            TextItem("-", x=400.0, y=650.0),
            TextItem("Page 2", x=290.0, y=40.0, size=9.0),
        ]
    )
    return build_pdf([page1, page2])


def single_page_pdf(text: str = "Hello LEDGER") -> bytes:
    """The smallest useful fixture: one page, one line of text."""
    return build_pdf([PageSpec(items=[TextItem(text, x=72.0, y=700.0)])])


def empty_text_pdf() -> bytes:
    """A page with no text layer at all, standing in for a scanned page."""
    return build_pdf([PageSpec(items=[])])


def large_pdf(min_bytes: int) -> bytes:
    """A genuinely large single-page PDF, for exercising the size limit.

    Real content rather than padding after ``%%EOF``, so the document stays
    valid and the test measures what the limit is actually for.
    """
    line = "Total revenue increased in the period under review. " * 2
    items: list[TextItem] = []
    # Roughly 100 bytes of content stream per item; overshoot, then trim.
    estimated = max(1, min_bytes // 100)
    for index in range(estimated):
        items.append(TextItem(line, x=72.0, y=float(700 - (index % 600)), size=9.0))

    data = build_pdf([PageSpec(items=items)])
    while len(data) < min_bytes:
        items.extend(items[: max(1, len(items) // 2)])
        data = build_pdf([PageSpec(items=items)])
    return data


__all__ = [
    "PAGE_HEIGHT",
    "PAGE_WIDTH",
    "PageSpec",
    "TextItem",
    "build_pdf",
    "empty_text_pdf",
    "simple_report_pdf",
    "single_page_pdf",
]
