"""The deep-learning backend: Docling's DocLayNet layout model + TableFormer.

This is the backend the project brief's requirement is actually about. Two
models do the work that no heuristic can:

* a **DocLayNet-trained layout model** classifies every region on the page —
  title, section header, text, list item, table, caption, footnote, page
  furniture — which is what gives us headings and reliable furniture removal;
* **TableFormer** recovers table *structure*: the cell grid, row and column
  spans, and which cells are headers. Tables are the heart of TAT-DQA, and
  nothing short of a structure model gets a financial statement's grid right.

Glyphs come from the PDF's own text layer on born-digital pages and from a
real OCR engine on scanned ones. That is a deliberate choice, not a shortcut:
force-OCR'ing a page whose text layer is already character-exact can only
introduce errors, and the errors would land in precisely the figures the
answer validator checks. ``force_ocr=True`` overrides it for the cases where
the classifier misjudges a page, and for demonstrating the OCR path.

Everything docling-specific is confined to this module. What leaves is
:class:`~app.backends.base.RawParse`, identical in shape to what the
text-layer backend produces, so nothing downstream can tell which model ran.

The mapping functions below take duck-typed arguments and import nothing from
docling, which is what lets the whole mapping layer be tested on CPU with no
model present. Only :class:`DoclingBackend` touches the library, and only
inside its methods.
"""

from __future__ import annotations

import io
from typing import Any, Iterable

from app.backends.base import RawBBox, RawBlock, RawCell, RawPage, RawParse, RawTable
from app.ingestion.render import PageGeometry, PdfDocument
from app.processing.typography import TypographyIndex
from ledger_doc_contract.v1.enums import BlockType, ExtractionSource

#: DocLayNet / docling labels mapped onto the contract's block taxonomy.
#:
#: Keyed on the label's string value rather than on the ``DocItemLabel`` enum
#: so this table can be read, and tested, without docling installed — and so a
#: docling release that adds a label cannot raise an ImportError here.
LABEL_MAP: dict[str, BlockType] = {
    "title": BlockType.TITLE,
    "section_header": BlockType.SECTION_HEADER,
    "text": BlockType.TEXT,
    "paragraph": BlockType.TEXT,
    "reference": BlockType.TEXT,
    "code": BlockType.TEXT,
    "list_item": BlockType.LIST_ITEM,
    "table": BlockType.TABLE,
    "document_index": BlockType.TABLE,
    "caption": BlockType.CAPTION,
    "footnote": BlockType.FOOTNOTE,
    "picture": BlockType.FIGURE,
    "chart": BlockType.FIGURE,
    "formula": BlockType.FORMULA,
    "page_header": BlockType.PAGE_HEADER,
    "page_footer": BlockType.PAGE_FOOTER,
}


def map_label(label: Any) -> BlockType:
    """Map a docling label onto a contract block type.

    An unrecognised label becomes :attr:`BlockType.TEXT` rather than being
    dropped. A future docling release that introduces a label we have never
    seen should cost the consumer a slightly wrong ``content_type``, never the
    text itself — a silently missing paragraph is an unanswerable question.
    """
    name = getattr(label, "value", label)
    return LABEL_MAP.get(str(name).lower(), BlockType.TEXT)


def is_bottom_left(bbox: Any) -> bool:
    """Whether a docling box uses the PDF's bottom-left origin.

    Docling reports boxes in either origin depending on how the page was
    processed, and carries which one on the box itself. Reading it is the
    difference between a citation highlight landing on the right paragraph and
    landing on its mirror image.
    """
    origin = getattr(bbox, "coord_origin", None)
    if origin is None:
        return False
    return str(getattr(origin, "value", origin)).upper().startswith("BOTTOM")


def to_raw_bbox(
    bbox: Any,
    *,
    page_height: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> RawBBox | None:
    """Convert a docling box to the contract's convention.

    The contract promises one geometry for every backend: PDF points,
    top-left origin, y increasing downward. Two corrections get us there —
    a flip when the box is bottom-left, and a scale when docling's page size
    differs from the PDF's own.

    ``page_height`` must be docling's height for the page, since the flip
    happens in docling's space and the scale is applied after it.
    """
    if bbox is None:
        return None

    try:
        left = float(bbox.l)
        top = float(bbox.t)
        right = float(bbox.r)
        bottom = float(bbox.b)
    except (AttributeError, TypeError, ValueError):
        return None

    if is_bottom_left(bbox):
        y0 = page_height - max(top, bottom)
        y1 = page_height - min(top, bottom)
    else:
        y0, y1 = min(top, bottom), max(top, bottom)

    return RawBBox(
        x0=round(min(left, right) * scale_x, 2),
        y0=round(y0 * scale_y, 2),
        x1=round(max(left, right) * scale_x, 2),
        y1=round(y1 * scale_y, 2),
    )


def _span(cell: Any, start_attr: str, end_attr: str, fallback: str) -> int:
    """A cell's span, preferred from its offsets.

    Offsets are derived from the grid itself, so they stay consistent with the
    row and column indices we place the cell at. The explicit span attribute
    is only a fallback, and a span of zero — which some versions report for a
    degenerate cell — is corrected to one so the grid keeps its shape.
    """
    start = getattr(cell, start_attr, None)
    end = getattr(cell, end_attr, None)
    if isinstance(start, int) and isinstance(end, int) and end > start:
        return end - start
    value = getattr(cell, fallback, 1)
    return value if isinstance(value, int) and value > 0 else 1


def header_extent(cells: Iterable[Any], *, n_rows: int, n_cols: int) -> tuple[int, int]:
    """Count the leading header rows and header columns.

    A row counts as a header row only when *every* cell starting in it is
    flagged as a column header, and counting stops at the first row that is
    not. TableFormer occasionally marks a stray body cell as a header; taking
    the leading run rather than the total keeps one such cell from declaring
    half the statement to be headings.
    """
    column_header_rows: dict[int, list[bool]] = {}
    row_header_cols: dict[int, list[bool]] = {}

    for cell in cells:
        row = getattr(cell, "start_row_offset_idx", None)
        col = getattr(cell, "start_col_offset_idx", None)
        if isinstance(row, int):
            column_header_rows.setdefault(row, []).append(
                bool(getattr(cell, "column_header", False))
            )
        if isinstance(col, int):
            row_header_cols.setdefault(col, []).append(
                bool(getattr(cell, "row_header", False))
            )

    header_rows = 0
    for row in range(n_rows):
        flags = column_header_rows.get(row)
        if not flags or not all(flags):
            break
        header_rows += 1

    header_cols = 0
    for col in range(n_cols):
        flags = row_header_cols.get(col)
        if not flags or not all(flags):
            break
        header_cols += 1

    return header_rows, header_cols


def to_raw_table(
    table_data: Any,
    *,
    caption: str | None = None,
    page_height: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    confidence: float | None = None,
) -> RawTable:
    """Convert TableFormer's grid into the backend-neutral table.

    Cells are placed at their start offsets and carry their spans, which is
    what lets a merged 'Year ended December 31' header covering three year
    columns resolve correctly on lookup instead of collapsing into one.
    """
    cells_in = list(getattr(table_data, "table_cells", ()) or ())

    n_rows = getattr(table_data, "num_rows", 0) or 0
    n_cols = getattr(table_data, "num_cols", 0) or 0
    if not n_rows or not n_cols:
        # Some versions leave the dimensions unset. Deriving them from the
        # cells is always possible and always consistent with them.
        n_rows = max(
            (getattr(c, "end_row_offset_idx", 0) or 0 for c in cells_in), default=0
        )
        n_cols = max(
            (getattr(c, "end_col_offset_idx", 0) or 0 for c in cells_in), default=0
        )

    header_rows, header_cols = header_extent(cells_in, n_rows=n_rows, n_cols=n_cols)

    cells: list[RawCell] = []
    for cell in cells_in:
        row = getattr(cell, "start_row_offset_idx", None)
        col = getattr(cell, "start_col_offset_idx", None)
        if not isinstance(row, int) or not isinstance(col, int):
            continue
        cells.append(
            RawCell(
                row=row,
                col=col,
                text=str(getattr(cell, "text", "") or ""),
                row_span=_span(cell, "start_row_offset_idx", "end_row_offset_idx", "row_span"),
                col_span=_span(cell, "start_col_offset_idx", "end_col_offset_idx", "col_span"),
                is_header=bool(getattr(cell, "column_header", False))
                or bool(getattr(cell, "row_header", False)),
                bbox=to_raw_bbox(
                    getattr(cell, "bbox", None),
                    page_height=page_height,
                    scale_x=scale_x,
                    scale_y=scale_y,
                ),
            )
        )

    # Sorted so two runs over the same PDF emit cells in the same order. The
    # contract promises a byte-identical document body, and a set-ordered
    # cell list would quietly break that.
    cells.sort(key=lambda c: (c.row, c.col))

    return RawTable(
        n_rows=n_rows,
        n_cols=n_cols,
        cells=tuple(cells),
        header_rows=header_rows,
        header_cols=header_cols,
        caption=(caption.strip() or None) if caption else None,
        confidence=confidence,
    )


class DoclingBackend:
    """Docling's layout and table models behind the standard backend seam."""

    name = "docling"

    def __init__(
        self,
        *,
        table_mode: str = "accurate",
        ocr_enabled: bool = True,
        artifacts_path: str | None = None,
        num_threads: int = 4,
    ) -> None:
        self._table_mode = table_mode.lower()
        self._ocr_enabled = ocr_enabled
        self._artifacts_path = artifacts_path
        self._num_threads = num_threads
        # One converter per pipeline shape. force_ocr is a different pipeline,
        # not a per-call flag, so the two are built and kept separately rather
        # than rebuilt — model loading is the expensive part of a run.
        self._converters: dict[bool, Any] = {}

    # -- provenance ---------------------------------------------------------

    @property
    def version(self) -> str:
        from importlib.metadata import PackageNotFoundError, version

        parts = []
        for package in ("docling", "docling-ibm-models"):
            try:
                parts.append(f"{package} {version(package)}")
            except PackageNotFoundError:  # pragma: no cover - reporting only
                continue
        return " / ".join(parts) or "docling unknown"

    @property
    def model_versions(self) -> dict[str, str]:
        """Which models produced a parse.

        Recorded on every document because a backend comparison is only
        meaningful if each result says what actually ran. 'docling accurate'
        and 'docling fast' are different systems and score differently.
        """
        return {
            "layout": self._layout_model(),
            "table": f"tableformer-{self._table_mode}",
            "ocr": self._ocr_engine() if self._ocr_enabled else "disabled",
        }

    @staticmethod
    def _defaults() -> Any:
        """Docling's default pipeline options, or None if unavailable."""
        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            return PdfPipelineOptions()
        except Exception:  # pragma: no cover - reporting only
            return None

    def _layout_model(self) -> str:
        """The layout model docling would actually load.

        Named exactly rather than described generically. The field exists so a
        backend comparison can say what produced each number, and
        'docling-layout' would not distinguish two releases that score
        differently.
        """
        options = self._defaults()
        spec = getattr(getattr(options, "layout_options", None), "model_spec", None)
        repo = getattr(spec, "repo_id", None)
        revision = getattr(spec, "revision", None)
        if repo:
            return f"{repo}@{revision}" if revision else str(repo)
        return "docling-layout (DocLayNet)"

    def _ocr_engine(self) -> str:
        """The OCR engine docling would actually use.

        Read from the pipeline rather than hard-coded: docling picks an engine
        from what is installed, and a provenance record naming an engine that
        did not run is worse than one naming none.
        """
        options = self._defaults()
        kind = getattr(getattr(options, "ocr_options", None), "kind", None)
        return str(kind) if kind else "unknown"

    # -- lifecycle ----------------------------------------------------------

    def is_ready(self) -> bool:
        """True once the default pipeline's models are loaded.

        Reported honestly so ``GET /ready`` does not invite traffic into a
        service that would then spend a minute downloading weights.
        """
        return False in self._converters

    def warm(self) -> None:
        """Load the models. Safe to call repeatedly."""
        self._converter(force_ocr=False)

    def _converter(self, *, force_ocr: bool) -> Any:
        if force_ocr in self._converters:
            return self._converters[force_ocr]

        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableFormerMode,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_table_structure = True
        options.table_structure_options.mode = (
            TableFormerMode.FAST
            if self._table_mode == "fast"
            else TableFormerMode.ACCURATE
        )
        # Matching recovered cells back to the PDF's own glyphs is what keeps
        # a born-digital table's figures character-exact instead of re-read.
        options.table_structure_options.do_cell_matching = True
        options.do_ocr = self._ocr_enabled or force_ocr
        options.generate_page_images = False

        if force_ocr:
            options.ocr_options.force_full_page_ocr = True
        if self._artifacts_path:
            # Set for an air-gapped deployment: models are read from disk
            # rather than fetched, so a run never depends on the network.
            options.artifacts_path = self._artifacts_path
        try:
            options.accelerator_options.num_threads = self._num_threads
        except AttributeError:  # pragma: no cover - older docling
            pass

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        try:
            converter.initialize_pipeline(InputFormat.PDF)
        except Exception:  # pragma: no cover - warming is best effort
            # An eager warm is an optimisation. If this docling version has no
            # such hook, the first parse pays for the load instead.
            pass

        self._converters[force_ocr] = converter
        return converter

    # -- parsing ------------------------------------------------------------

    def parse(self, data: bytes, *, force_ocr: bool = False) -> RawParse:
        from docling.datamodel.base_models import DocumentStream

        converter = self._converter(force_ocr=force_ocr)

        result = converter.convert(
            DocumentStream(name="document.pdf", stream=io.BytesIO(data))
        )
        document = result.document

        warnings: list[tuple[str, str, int | None]] = []
        pages: list[RawPage] = []

        # pypdfium2 is the authority on how many pages the PDF has and how big
        # they are. Iterating it rather than docling's page map means a page
        # docling dropped is still reported — as an empty, failed page with a
        # warning, which a consumer can act on — instead of vanishing and
        # silently shortening the document.
        with PdfDocument(data) as pdf:
            geometries = {
                number: pdf.geometry(number)
                for number in range(1, pdf.page_count + 1)
            }
            blocks_by_page = self._collect(document, warnings, geometries)

            for page_number, geometry in geometries.items():
                try:
                    chars = pdf.text_chars(page_number)
                except Exception:  # pragma: no cover - defensive
                    chars = []

                blocks = blocks_by_page.get(page_number, [])

                # Docling reports no type size, and heading levels need one.
                # Measuring it from the text layer inside each block the model
                # found is the join between the two: the model says what a
                # region is, the glyphs say how it looks.
                index = TypographyIndex(chars)
                blocks = [self._with_typography(block, index) for block in blocks]

                if force_ocr:
                    source = ExtractionSource.OCR
                elif chars:
                    source = ExtractionSource.TEXT_LAYER
                elif self._ocr_enabled:
                    source = ExtractionSource.OCR
                    warnings.append(
                        (
                            "SCANNED_PAGE_OCR_FALLBACK",
                            f"Page {page_number} has no text layer; its text "
                            f"was recovered by OCR.",
                            page_number,
                        )
                    )
                else:
                    source = ExtractionSource.FAILED
                    warnings.append(
                        (
                            "PAGE_EXTRACTION_FAILED",
                            f"Page {page_number} has no text layer and OCR is "
                            f"disabled, so no text could be recovered.",
                            page_number,
                        )
                    )

                if not blocks and source is not ExtractionSource.FAILED:
                    warnings.append(
                        (
                            "PAGE_EXTRACTION_FAILED",
                            f"Page {page_number} yielded no layout blocks.",
                            page_number,
                        )
                    )

                pages.append(
                    RawPage(
                        page_number=page_number,
                        width=geometry.width,
                        height=geometry.height,
                        rotation=geometry.rotation,
                        extraction_source=source,
                        ocr_confidence=self._page_confidence(result, page_number),
                        blocks=tuple(blocks),
                    )
                )

        return RawParse(pages=tuple(pages), warnings=tuple(warnings))

    # -- internals ----------------------------------------------------------

    def _collect(
        self,
        document: Any,
        warnings: list[tuple[str, str, int | None]],
        geometries: dict[int, PageGeometry],
    ) -> dict[int, list[RawBlock]]:
        """Walk the converted document and group its items by page."""
        by_page: dict[int, list[RawBlock]] = {}

        page_sizes = self._page_sizes(document, geometries)

        unplaceable = 0

        for item, _level in document.iterate_items():
            provenance = list(getattr(item, "prov", ()) or ())
            if not provenance:
                # An item with no provenance cannot be put on a page or given
                # a box, so it can never be cited and is dropped. Counted
                # rather than reported one by one: a document with forty such
                # items would otherwise bury its real warnings under forty
                # identical lines, and the consumer needs the number, not the
                # repetition.
                unplaceable += 1
                continue

            prov = provenance[0]
            page_number = int(getattr(prov, "page_no", 0) or 0)
            if page_number <= 0:
                continue

            source_height, scale_x, scale_y = page_sizes.get(
                page_number, (0.0, 1.0, 1.0)
            )
            bbox = to_raw_bbox(
                getattr(prov, "bbox", None),
                page_height=source_height,
                scale_x=scale_x,
                scale_y=scale_y,
            )

            table_data = getattr(item, "data", None)
            has_grid = table_data is not None and hasattr(table_data, "table_cells")

            if has_grid:
                raw_table = to_raw_table(
                    table_data,
                    caption=self._caption(item, document),
                    page_height=source_height,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
                if raw_table.n_rows == 0 or raw_table.n_cols == 0:
                    warnings.append(
                        (
                            "TABLE_STRUCTURE_UNCERTAIN",
                            f"A table on page {page_number} was detected but no "
                            f"cell grid could be recovered from it.",
                            page_number,
                        )
                    )
                block = RawBlock(
                    type=BlockType.TABLE,
                    text="",
                    bbox=bbox,
                    table=raw_table,
                )
            else:
                kind = map_label(getattr(item, "label", None))
                text = str(getattr(item, "text", "") or "").strip()
                if not text and kind is not BlockType.FIGURE:
                    continue
                # A figure legitimately has no text. It is still emitted, with
                # its box, so a page's reading order does not silently close
                # over the gap where a chart sits — a consumer can see that
                # something unread is there rather than infer continuity that
                # was never on the page.
                block = RawBlock(type=kind, text=text, bbox=bbox)

            by_page.setdefault(page_number, []).append(block)

        if unplaceable:
            warnings.append(
                (
                    "PAGE_EXTRACTION_FAILED",
                    f"{unplaceable} item(s) carried no page provenance and "
                    f"were dropped; their text is not in this document.",
                    None,
                )
            )

        return by_page

    @staticmethod
    def _page_sizes(
        document: Any, geometries: dict[int, PageGeometry]
    ) -> dict[int, tuple[float, float, float]]:
        """Per page: docling's height, and the scale from it onto PDF points.

        Docling normally reports page sizes in points already, making both
        scales 1.0. Deriving them from pypdfium2's geometry rather than
        assuming that costs nothing and means the contract's "PDF points"
        promise does not rest on a third-party default staying put.

        A page docling did not report falls back to the PDF's own size with no
        scaling, which is the identity — right for the common case and never
        worse than inventing a factor.
        """
        sizes: dict[int, tuple[float, float, float]] = {}

        for number, geometry in geometries.items():
            sizes[number] = (geometry.height, 1.0, 1.0)

        for number, page in (getattr(document, "pages", {}) or {}).items():
            number = int(number)
            geometry = geometries.get(number)
            if geometry is None:
                continue
            size = getattr(page, "size", None)
            width = float(getattr(size, "width", 0.0) or 0.0)
            height = float(getattr(size, "height", 0.0) or 0.0)
            if width <= 0 or height <= 0:
                continue
            sizes[number] = (
                height,
                geometry.width / width,
                geometry.height / height,
            )

        return sizes

    @staticmethod
    def _caption(item: Any, document: Any) -> str | None:
        try:
            text = item.caption_text(document)
        except Exception:  # pragma: no cover - captions are optional
            return None
        text = str(text or "").strip()
        return text or None

    @staticmethod
    def _with_typography(block: RawBlock, index: TypographyIndex) -> RawBlock:
        """Attach measured type to a block, where the text layer has any."""
        if block.type is BlockType.TABLE:
            # A table's type size is meaningless — it is measured across
            # headers, labels, and figures at once — and it is never used for
            # level assignment. Measuring it would only add noise.
            return block

        measured = index.probe(block.bbox)
        if not measured.measured:
            return block

        return RawBlock(
            type=block.type,
            text=block.text,
            bbox=block.bbox,
            confidence=block.confidence,
            table=block.table,
            font_size=measured.font_size,
            is_bold=measured.is_bold,
        )

    @staticmethod
    def _page_confidence(result: Any, page_number: int) -> float | None:
        """Docling's OCR confidence for a page, when it reports one.

        Returns ``None`` rather than a fabricated 1.0 when nothing was
        measured — the contract distinguishes 'not measured' from 'certain',
        and conflating them is what makes a quality gate useless.
        """
        report = getattr(result, "confidence", None)
        pages = getattr(report, "pages", None)
        if not pages:
            return None
        page = pages.get(page_number) if hasattr(pages, "get") else None
        score = getattr(page, "ocr_score", None)
        try:
            value = float(score)
        except (TypeError, ValueError):
            return None
        # Docling uses NaN for "no measurement taken".
        if value != value or not 0.0 <= value <= 1.0:
            return None
        return round(value, 4)


__all__ = [
    "LABEL_MAP",
    "DoclingBackend",
    "header_extent",
    "is_bottom_left",
    "map_label",
    "to_raw_bbox",
    "to_raw_table",
]
