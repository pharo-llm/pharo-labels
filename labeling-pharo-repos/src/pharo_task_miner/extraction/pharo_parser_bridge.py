"""Parser backends: the built-in Python parser and an optional Pharo bridge.

The ``pharo`` backend sends method sources (never repository code to execute)
to a Pharo image that parses them with ``RBParser`` via the ``ASTExporter``
package in ``pharo/ASTExporter``. The image is copied to a temporary directory
for every batch so the researcher's image is never modified. Both backends
return the same AST vocabulary, so downstream analysis is backend-agnostic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .smalltalk_parser import ParseError, parse_method

PYTHON_BACKEND_VERSION = "python-st-parser/1.0"


class ParserBackend:
    name = "abstract"
    version = "0"

    def parse_batch(self, items: list[tuple[str, str]]) -> dict[str, tuple[dict | None, str | None]]:
        """Parse ``[(key, method_source)]`` → ``{key: (ast, error)}``."""
        raise NotImplementedError


class PythonBackend(ParserBackend):
    name = "python"
    version = PYTHON_BACKEND_VERSION

    def parse_batch(self, items):
        out = {}
        for key, source in items:
            try:
                out[key] = (parse_method(source), None)
            except ParseError as e:
                out[key] = (None, str(e))
            except RecursionError:
                out[key] = (None, "nesting too deep for the Python parser")
        return out


class PharoBackend(ParserBackend):
    name = "pharo"

    def __init__(self, vm: str, image: str, exporter_dir: str, timeout: int = 900):
        if not vm or not image:
            raise ValueError("parsing.pharo_vm and parsing.pharo_image must be configured for the pharo backend")
        self.vm, self.image = vm, Path(image)
        self.exporter_dir = Path(exporter_dir).resolve()
        self.timeout = timeout
        self.version = f"pharo-rbparser/{self.image.name}"

    def parse_batch(self, items):
        with tempfile.TemporaryDirectory(prefix="ptm-pharo-") as tmp:
            tmp = Path(tmp)
            image = tmp / self.image.name
            shutil.copy2(self.image, image)
            changes = self.image.with_suffix(".changes")
            if changes.exists():
                shutil.copy2(changes, image.with_suffix(".changes"))
            for sources in self.image.parent.glob("*.sources"):
                shutil.copy2(sources, tmp / sources.name)
            inp, outp = tmp / "methods.jsonl", tmp / "asts.jsonl"
            with inp.open("w", encoding="utf-8") as f:
                for key, source in items:
                    f.write(json.dumps({"id": key, "source": source}) + "\n")
            script = tmp / "export.st"
            script.write_text(
                "Smalltalk at: #ASTExporter ifAbsent: [\n"
                f"  Metacello new baseline: 'ASTExporter'; repository: 'tonel://{self.exporter_dir}'; load ].\n"
                f"(Smalltalk at: #ASTExporter) exportFrom: '{inp}' to: '{outp}'.\n"
                "Smalltalk snapshot: false andQuit: true.\n",
                encoding="utf-8",
            )
            env = dict(os.environ)
            proc = subprocess.run([self.vm, "--headless", str(image), "st", str(script)],
                                  cwd=tmp, capture_output=True, text=True, timeout=self.timeout, env=env)
            if proc.returncode != 0 or not outp.exists():
                raise RuntimeError(f"Pharo parser failed (exit {proc.returncode}): {proc.stderr[-2000:]}")
            out = {}
            for line in outp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    out[rec["id"]] = (rec.get("ast"), rec.get("error"))
            for key, _ in items:
                out.setdefault(key, (None, "no result returned by Pharo exporter"))
            return out


def make_backend(parsing_cfg: dict, project_root: Path) -> ParserBackend:
    backend = (parsing_cfg.get("backend") or "python").lower()
    if backend == "python":
        return PythonBackend()
    if backend == "pharo":
        return PharoBackend(parsing_cfg.get("pharo_vm"), parsing_cfg.get("pharo_image"),
                            str(project_root / "pharo" / "ASTExporter"), int(parsing_cfg.get("pharo_timeout", 900)))
    raise ValueError(f"Unknown parser backend {backend!r}")
