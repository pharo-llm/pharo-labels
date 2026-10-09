"""Extract method-level records (with provenance and AST facts) from a checkout.

Phase 1 reads every source container and builds the repository class table.
Phase 2 parses method sources with the configured backend and derives the
syntactic facts required by the specification (message sends, blocks,
literals, returns, assignments, referenced globals, variable scopes).
Nothing from the repository is executed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re

from .tonel_reader import ClassDefinition, FileResult, MethodSource, read_source_file
from .smalltalk_parser import iter_nodes, count_nodes, PSEUDO_VARIABLES
from .pharo_parser_bridge import ParserBackend

SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv"}
TEST_SUPERCLASSES = {"TestCase", "ParametrizedTestCase", "TestResource", "BlTestCase", "SpTest", "ClyBrowserTestCase"}
_WS = re.compile(r"\s+")
# Messages the Pharo compiler inlines: their literal block arguments are not
# real closures (a ^ inside them is an ordinary method return).
INLINED_SELECTORS = {"ifTrue:", "ifFalse:", "ifTrue:ifFalse:", "ifFalse:ifTrue:", "and:", "or:", "whileTrue:",
                     "whileFalse:", "whileTrue", "whileFalse", "to:do:", "to:by:do:", "timesRepeat:", "ifNil:",
                     "ifNotNil:", "ifNil:ifNotNil:", "ifNotNil:ifNil:", "repeat"}


def source_hash(source: str) -> str:
    """Whitespace-insensitive content fingerprint of a method's source."""
    return hashlib.sha1(_WS.sub(" ", source).strip().encode("utf-8")).hexdigest()


def method_key(repo: str, commit: str | None, cls: str, side: str, selector: str, path: str) -> str:
    raw = "|".join([repo, commit or "", cls, side, selector, path])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


@dataclass
class RepositoryExtraction:
    repository: str
    commit_sha: str | None
    files: list[dict] = field(default_factory=list)
    classes: list[dict] = field(default_factory=list)
    methods: list[dict] = field(default_factory=list)
    packages: dict = field(default_factory=dict)
    baseline_dependencies: list[dict] = field(default_factory=list)
    parser_backend: str = ""


def iter_source_files(root: Path, exclude_globs: list[str] | None = None):
    exclude_globs = exclude_globs or []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            full = Path(dirpath) / fn
            rel = full.relative_to(root).as_posix()
            if any(Path(rel).match(g) for g in exclude_globs):
                continue
            yield full, rel


def _class_table(file_results: list[FileResult]) -> dict[str, ClassDefinition]:
    table: dict[str, ClassDefinition] = {}
    for fr in file_results:
        for c in fr.classes:
            if c.kind == "extension":
                continue
            prev = table.get(c.name)
            if prev is None or (not prev.superclass and c.superclass):
                table[c.name] = c
    return table


def _ancestors(name: str, table: dict[str, ClassDefinition]) -> list[ClassDefinition]:
    out, seen = [], set()
    cur = table.get(name)
    while cur is not None and cur.name not in seen:
        seen.add(cur.name)
        out.append(cur)
        cur = table.get(cur.superclass) if cur.superclass else None
    return out


def root_external_superclass(name: str, table: dict[str, ClassDefinition]) -> str | None:
    chain = _ancestors(name, table)
    return chain[-1].superclass if chain else None


def is_test_class(name: str, table: dict[str, ClassDefinition], package: str | None) -> bool:
    for c in _ancestors(name, table):
        if c.superclass in TEST_SUPERCLASSES or (c.superclass or "").endswith("TestCase"):
            return True
    if name.endswith(("Test", "Tests", "TestCase")):
        return True
    return bool(package and re.search(r"(^|[-_.])Tests?($|[-_.])", package))


class Scope:
    """Lexical scope for variable classification within one method."""

    def __init__(self, cls: str, side: str, table: dict[str, ClassDefinition]):
        self.inst_vars, self.class_vars, self.pools = set(), set(), set()
        for c in _ancestors(cls, table):
            self.inst_vars.update(c.class_instance_variables if side == "class" else c.instance_variables)
            self.class_vars.update(c.class_variables)
            self.pools.update(c.pools)
        self.frames: list[tuple[str, set[str]]] = []

    def push(self, kind: str, names):
        self.frames.append((kind, set(names)))

    def pop(self):
        self.frames.pop()

    def classify(self, name: str) -> str:
        if name in PSEUDO_VARIABLES:
            return "pseudo"
        for kind, names in reversed(self.frames):
            if name in names:
                return kind
        if name in self.inst_vars:
            return "instance_variable"
        if name in self.class_vars:
            return "class_variable"
        if name[:1].isupper():
            return "global"
        return "unresolved"  # e.g. inherited from a superclass outside the repository


def _literal_summary(node: dict) -> dict:
    v = str(node.get("value"))
    return {"kind": node.get("kind"), "value": v if len(v) <= 80 else v[:77] + "..."}


def _expr_type(node: dict | None) -> str:
    if not node:
        return "none"
    t = node.get("type")
    if t == "Literal":
        return "literal:" + str(node.get("kind"))
    if t == "Variable":
        return "variable:" + node.get("name", "")
    if t == "Message":
        return "message:" + node.get("selector", "")
    return t.lower() if t else "unknown"


def ast_facts(ast: dict, cls: str, side: str, table: dict[str, ClassDefinition]) -> dict:
    """Walk an AST with scope tracking and return the specification's fields."""
    scope = Scope(cls, side, table)
    scope.push("argument", ast.get("arguments", []))
    sends, blocks, literals, returns, assignments = [], [], [], [], []
    globals_, var_reads = [], defaultdict(set)
    temporaries = list(ast.get("body", {}).get("temporaries", []))
    block_args_all, block_temps_all = [], []
    max_block_depth = 0

    def visit(node, block_depth, in_cascade=False, cascade_receiver=None, inlined=False):
        nonlocal max_block_depth
        if not isinstance(node, dict):
            return
        t = node.get("type")
        if t == "Sequence":
            scope.push("temporary", node.get("temporaries", []))
            for s in node.get("statements", []):
                visit(s, block_depth)
            scope.pop()
        elif t == "Block":
            args = node.get("arguments", [])
            temps = node.get("body", {}).get("temporaries", [])
            block_args_all.extend(args)
            block_temps_all.extend(temps)
            depth = block_depth if inlined else block_depth + 1
            max_block_depth = max(max_block_depth, depth)
            stmts = node.get("body", {}).get("statements", [])
            has_ret = any(n.get("type") == "Return" for n, _, _, _ in iter_nodes(node.get("body")))
            blocks.append({"line": node.get("line"), "n_arguments": len(args), "n_temporaries": len(temps),
                           "depth": depth, "n_statements": len(stmts), "inlined": inlined,
                           "contains_return": has_ret})
            scope.push("block_argument", args)
            visit(node.get("body"), depth)
            scope.pop()
        elif t == "Return":
            returns.append({"line": node.get("line"), "in_block": block_depth > 0, "value_type": _expr_type(node.get("value"))})
            visit(node.get("value"), block_depth)
        elif t == "Assignment":
            name = node.get("variable")
            assignments.append({"variable": name, "scope": scope.classify(name), "line": node.get("line"),
                                "value_type": _expr_type(node.get("value")), "in_block": block_depth > 0})
            visit(node.get("value"), block_depth)
        elif t == "Cascade":
            visit(node.get("receiver"), block_depth)
            for m in node.get("messages", []):
                visit(m, block_depth, in_cascade=True, cascade_receiver=node.get("receiver"))
        elif t == "Message":
            receiver = node.get("receiver") or {}
            if receiver.get("type") == "CascadeReceiver":
                eff = cascade_receiver or {}
            else:
                eff = receiver
                visit(receiver, block_depth, in_cascade, cascade_receiver,
                      inlined=node.get("selector") in INLINED_SELECTORS and receiver.get("type") == "Block")
            rkind = eff.get("type", "unknown")
            rname = None
            if rkind == "Variable":
                rname = eff.get("name")
                rkind = scope.classify(rname)
                if rname in ("self", "super", "thisContext"):
                    rkind = rname
            elif rkind == "Literal":
                rname = eff.get("kind")
                rkind = "literal"
            elif rkind == "Block":
                rkind = "block"
            elif rkind == "Message":
                rname = eff.get("selector")
                rkind = "message_result"
            elif rkind == "Cascade":
                rkind = "cascade_result"
            elif rkind == "DynamicArray":
                rkind = "dynamic_array"
            sends.append({
                "selector": node.get("selector"), "kind": node.get("kind"), "line": node.get("line"),
                "receiver_kind": rkind, "receiver_name": rname, "in_cascade": in_cascade or receiver.get("type") == "CascadeReceiver",
                "block_depth": block_depth,
                "argument_types": [_expr_type(a) for a in node.get("arguments", [])],
            })
            inline_args = node.get("selector") in INLINED_SELECTORS
            for a in node.get("arguments", []):
                visit(a, block_depth, inlined=inline_args and a.get("type") == "Block")
        elif t == "Variable":
            name = node.get("name")
            kind = scope.classify(name)
            var_reads[kind].add(name)
            if kind == "global":
                globals_.append({"name": name, "line": node.get("line")})
        elif t == "Literal":
            literals.append(_literal_summary(node))
        elif t == "DynamicArray":
            for s in node.get("statements", []):
                visit(s, block_depth)

    visit(ast.get("body"), 0)
    top = (ast.get("body") or {}).get("statements", [])
    method_returns = [r for r in returns if not r["in_block"]]
    has_early_return = len(method_returns) > 1 or bool(method_returns and (not top or top[-1].get("type") != "Return"))
    return {
        "has_early_return": has_early_return,
        "temporaries": temporaries,
        "block_arguments": block_args_all,
        "block_temporaries": block_temps_all,
        "message_sends": sends,
        "blocks": blocks,
        "literals": literals,
        "returns": returns,
        "assignments": assignments,
        "referenced_globals": globals_,
        "variables_read": {k: sorted(v) for k, v in var_reads.items()},
        "pragmas": [{"selector": p.get("selector"), "arguments": p.get("arguments", [])} for p in ast.get("pragmas", [])],
        "n_ast_nodes": count_nodes(ast),
        "max_block_depth": max_block_depth,
    }


_REQUIRES = re.compile(r"'([^']+)'|#'([^']+)'|#([A-Za-z][\w-]*)")


def _string_args(node: dict) -> list[str]:
    if not isinstance(node, dict) or node.get("type") != "Literal":
        return []
    if node.get("kind") in ("string", "symbol"):
        return [str(node.get("value"))]
    if node.get("kind") == "array":
        return [a or b or c for a, b, c in _REQUIRES.findall(str(node.get("value")))]
    return []


def baseline_dependencies(methods: list[dict]) -> list[dict]:
    """Dependencies declared in BaselineOf/ConfigurationOf specifications."""
    out = []
    for m in methods:
        if not m.get("ast") or not re.match(r"(BaselineOf|ConfigurationOf)", m["class_name"]):
            continue
        current_package = None
        for node, _p, _d, _b in iter_nodes(m["ast"]):
            if node.get("type") != "Message":
                continue
            sel, args = node.get("selector"), node.get("arguments", [])
            if sel in ("baseline:with:", "project:with:", "baseline:", "project:", "configuration:with:") and args:
                for name in _string_args(args[0]):
                    out.append({"kind": "external_project", "name": name, "from": m["class_name"],
                                "selector": sel, "line": node.get("line")})
            elif sel in ("package:with:", "package:") and args:
                names = _string_args(args[0])
                current_package = names[0] if names else None
                for name in names:
                    out.append({"kind": "package", "name": name, "from": m["class_name"], "selector": sel, "line": node.get("line")})
            elif sel in ("requires:", "includes:") and args:
                for name in _string_args(args[0]):
                    out.append({"kind": "requires", "name": name, "package": current_package,
                                "from": m["class_name"], "selector": sel, "line": node.get("line")})
    return out


def extract_repository(root: Path, repository: str, commit_sha: str | None, backend: ParserBackend,
                       include_tests: bool = True, include_extensions: bool = True, export_ast: bool = True,
                       exclude_globs: list[str] | None = None, batch_size: int = 5000) -> RepositoryExtraction:
    res = RepositoryExtraction(repository, commit_sha, parser_backend=f"{backend.name}:{backend.version}")
    file_results: list[FileResult] = []
    for full, rel in iter_source_files(root, exclude_globs):
        fr = read_source_file(full, rel)
        if fr is not None:
            file_results.append(fr)
    table = _class_table(file_results)
    method_sources: list[MethodSource] = []
    for fr in file_results:
        res.files.append({"path": fr.path, "format": fr.format, "status": fr.status, "n_methods": len(fr.methods),
                          "n_classes": len(fr.classes), "package": fr.package, "warnings": fr.warnings,
                          "dialect_hints": fr.dialect_hints})
        method_sources.extend(fr.methods)
        for c in fr.classes:
            d = c.to_dict()
            d["package"] = c.package or fr.package or c.category
            d["is_test"] = is_test_class(c.name, table, d["package"]) if c.kind != "extension" else False
            d["external_root_superclass"] = root_external_superclass(c.name, table)
            res.classes.append(d)
        if fr.package:
            res.packages.setdefault(fr.package, {"name": fr.package, "paths": set()})["paths"].add(str(Path(fr.path).parent))
    # Package for methods without one: from the class definition.
    pkg_of_class = {c["name"]: c["package"] for c in res.classes if c.get("kind") != "extension"}
    pending = []
    for ms in method_sources:
        package = ms.package or pkg_of_class.get(ms.class_name)
        is_test = is_test_class(ms.class_name, table, package)
        if is_test and not include_tests:
            continue
        if ms.is_extension and not include_extensions:
            continue
        key = method_key(repository, commit_sha, ms.class_name, ms.side, ms.selector_hint, ms.source_path + f":{ms.line}")
        pending.append((key, ms, package, is_test))
    results = {}
    for i in range(0, len(pending), batch_size):
        chunk = pending[i:i + batch_size]
        results.update(backend.parse_batch([(k, ms.source) for k, ms, _, _ in chunk]))
    for key, ms, package, is_test in pending:
        ast, error = results.get(key, (None, "not parsed"))
        rec = {
            "method_key": key, "repository": repository, "commit_sha": commit_sha, "package": package,
            "class_name": ms.class_name, "side": ms.side, "selector": (ast or {}).get("selector") or ms.selector_hint,
            "protocol": ms.protocol, "source": ms.source, "source_path": ms.source_path, "line": ms.line,
            "container_format": ms.container_format, "is_extension": ms.is_extension, "in_trait": ms.in_trait,
            "is_test": is_test, "source_hash": source_hash(ms.source),
            "arguments": (ast or {}).get("arguments", []),
            "parse_status": "ok" if ast else "failed", "parse_errors": [error] if error else [],
            "parser_backend": res.parser_backend,
        }
        if ast:
            rec.update(ast_facts(ast, ms.class_name, ms.side, table))
            if rec["selector"] != ms.selector_hint and ms.container_format != "chunk":
                rec["parse_errors"].append(f"selector mismatch: container says {ms.selector_hint!r}")
        rec["ast"] = ast
        res.methods.append(rec)
    res.baseline_dependencies = baseline_dependencies(res.methods)
    if not export_ast:
        for m in res.methods:
            m["ast"] = None
    for p in res.packages.values():
        p["paths"] = sorted(p["paths"])
    return res
