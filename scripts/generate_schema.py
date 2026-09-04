"""Write the committed JSON Schema for the v1 contract.

Run after any deliberate contract change::

    python scripts/generate_schema.py

The committed file is the record of what consumers rely on. The stability test
compares the live models against it and fails on a removed field, so
regenerating it is how a change is consciously accepted rather than something
that happens silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "contracts"))

from ledger_doc_contract.v1 import ProcessedDocument  # noqa: E402

OUTPUT = ROOT / "contracts" / "ledger_doc_contract" / "v1" / "schema.json"


def main() -> int:
    schema = ProcessedDocument.model_json_schema()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(schema.get('$defs', {}))} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
