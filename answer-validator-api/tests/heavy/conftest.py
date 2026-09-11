"""Skip the whole heavy suite when the models are not available.

The rest of the test suite runs in about a second on CPU with no model and no
network, which is what makes it runnable on every commit. These tests are the
opposite: they load real weights and take minutes. Keeping them in their own
directory behind a skip means the fast suite stays fast and nobody has to
remember a flag to keep it that way.

Run them deliberately::

    pytest tests/heavy -q -m heavy
"""

from __future__ import annotations

import pytest

pytest.importorskip("docling", reason="the docling backend's models are not installed")

pytestmark = pytest.mark.heavy
