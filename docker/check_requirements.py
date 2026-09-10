"""Offline startup check for dependencies baked into a service image."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def iter_requirements(path: Path):
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            nested = line.split(maxsplit=1)[1]
            yield from iter_requirements((path.parent / nested).resolve())
            continue
        if line.startswith("-"):
            continue
        yield Requirement(line)


def main() -> int:
    failures: list[str] = []
    for filename in sys.argv[1:]:
        path = Path(filename)
        if not path.is_file():
            failures.append(f"missing requirements file: {path}")
            continue
        for requirement in iter_requirements(path):
            if requirement.marker and not requirement.marker.evaluate():
                continue
            name = canonicalize_name(requirement.name)
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                failures.append(f"{requirement}: not installed")
                continue
            if requirement.specifier and version not in requirement.specifier:
                failures.append(
                    f"{requirement}: installed version is {version}"
                )

    if failures:
        print("Container dependency verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print("Rebuild the affected image with docker compose build.", file=sys.stderr)
        return 1

    print("Container dependency verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
