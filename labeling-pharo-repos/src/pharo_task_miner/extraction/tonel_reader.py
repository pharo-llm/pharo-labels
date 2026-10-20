"""Readers for Pharo source containers: Tonel (primary), FileTree and chunk.

Readers split files into class definitions and *method source records*; they do
not parse method bodies (that is the parser backend's job). Every file is
classified, and structural problems are reported as warnings rather than being
skipped silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import re

from .smalltalk_parser import tokenize, line_of, ParseError, Token

TONEL_SUFFIXES = (".class.st", ".extension.st", ".trait.st")


@dataclass
class ClassDefinition:
    name: str
    kind: str  # class | trait | extension
    superclass: str | None = None
    package: str | None = None
    category: str | None = None
    tag: str | None = None
    instance_variables: list[str] = field(default_factory=list)
    class_variables: list[str] = field(default_factory=list)
    class_instance_variables: list[str] = field(default_factory=list)
    pools: list[str] = field(default_factory=list)
    traits: str | None = None
    class_traits: str | None = None
    comment: str | None = None
    source_path: str = ""
    line: int = 1
    raw_definition: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class MethodSource:
    class_name: str
    side: str  # instance | class
    selector_hint: str
    protocol: str | None
    source: str
    source_path: str
    line: int
    container_format: str
    package: str | None = None
    is_extension: bool = False
    in_trait: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class FileResult:
    path: str
    format: str  # tonel_class, tonel_extension, tonel_trait, tonel_package, filetree_method, filetree_class_metadata, ...
    status: str  # ok | partial | failed | unsupported | metadata
    classes: list[ClassDefinition] = field(default_factory=list)
    methods: list[MethodSource] = field(default_factory=list)
    package: str | None = None
    warnings: list[str] = field(default_factory=list)
    dialect_hints: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# STON-like maps used by Tonel headers and method metadata
# --------------------------------------------------------------------------

def _ston_value(tokens: list[Token], i: int, source: str):
    t = tokens[i]
    if t.kind == "symbol":
        v = t.value.lstrip("#")
        return (v[1:-1].replace("''", "'") if v.startswith("'") else v), i + 1
    if t.kind == "string":
        return t.value[1:-1].replace("''", "'"), i + 1
    if t.kind == "number":
        return t.value, i + 1
    if t.kind == "id" and t.value in ("true", "false", "nil"):
        return {"true": True, "false": False, "nil": None}[t.value], i + 1
    if t.value == "[":
        out, i = [], i + 1
        while tokens[i].value != "]":
            if tokens[i].value == ",":
                i += 1
                continue
            v, i = _ston_value(tokens, i, source)
            out.append(v)
        return out, i + 1
    if t.value == "{":
        return _ston_map(tokens, i, source)
    if t.kind == "id":  # bare identifiers occasionally appear (e.g. #type : variable)
        return t.value, i + 1
    raise ParseError(f"Unsupported STON value {t.value!r}", t.start)


def _ston_map(tokens: list[Token], i: int, source: str):
    if tokens[i].value != "{":
        raise ParseError("Expected '{'", tokens[i].start)
    out, i = {}, i + 1
    while tokens[i].value != "}":
        if tokens[i].value == ",":
            i += 1
            continue
        key_t = tokens[i]
        if key_t.kind == "symbol":
            key = key_t.value.lstrip("#").strip("'")
        elif key_t.kind == "string":
            key = key_t.value[1:-1]
        elif key_t.kind == "keyword":  # #key:value without spaces lexes oddly
            key = key_t.value[:-1]
        else:
            raise ParseError(f"Unsupported STON key {key_t.value!r}", key_t.start)
        key = key.rstrip(":")
        i += 1
        if tokens[i].value == ":":
            i += 1
        v, i = _ston_value(tokens, i, source)
        out[key] = v
    return out, i + 1


def _split_names(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [p for p in str(value).split() if p]


def _match_bracket(tokens: list[Token], i: int, open_v: str, close_v: str) -> int:
    """Return index of the token closing tokens[i] (which must be open_v)."""
    depth = 0
    for j in range(i, len(tokens)):
        v = tokens[j].value
        k = tokens[j].kind
        if k in ("bytearray",) and open_v == "[":
            depth += 1
            continue
        if v == open_v:
            depth += 1
        elif v == close_v:
            depth -= 1
            if depth == 0:
                return j
    raise ParseError(f"Unclosed {open_v!r}", tokens[i].start)


def _dedent_tonel_body(body: str) -> str:
    lines = body.split("\n")
    out = []
    for ln in lines:
        out.append(ln[1:] if ln.startswith("\t") else ln)
    text = "\n".join(out)
    stripped = text.rstrip()
    if stripped.endswith("$") and len(text) > len(stripped):
        stripped += text[len(stripped)]  # keep the character literal "$ "
    return stripped


# --------------------------------------------------------------------------
# Tonel
# --------------------------------------------------------------------------

def read_tonel(source: str, rel_path: str) -> FileResult:
    name = Path(rel_path).name
    if name == "package.st":
        fmt = "tonel_package"
    elif name.endswith(".extension.st"):
        fmt = "tonel_extension"
    elif name.endswith(".trait.st"):
        fmt = "tonel_trait"
    else:
        fmt = "tonel_class"
    result = FileResult(rel_path, fmt, "ok")
    try:
        tokens = tokenize(source, keep_comments=True)
    except ParseError as e:
        result.status = "failed"
        result.warnings.append(f"lexer: {e}")
        return result
    comment = None
    i = 0
    while i < len(tokens) and tokens[i].kind == "comment":
        comment = tokens[i].value[1:-1].replace('""', '"')
        i += 1
    tokens = [t for t in tokens if t.kind != "comment"]
    # re-locate i in filtered list: header starts at first non-comment token
    i = 0
    if not tokens:
        result.status = "failed"
        result.warnings.append("empty Tonel file")
        return result
    head = tokens[0]
    if head.kind != "id" or head.value not in ("Class", "Trait", "Extension", "Package"):
        result.status = "failed"
        result.warnings.append(f"not a Tonel definition header: {head.value!r}")
        return result
    try:
        header, i = _ston_map(tokens, 1, source)
    except (ParseError, IndexError) as e:
        result.status = "failed"
        result.warnings.append(f"header: {e}")
        return result
    if head.value == "Package":
        result.status = "metadata"
        result.package = str(header.get("name") or "") or None
        return result
    cname = str(header.get("name") or "")
    package = header.get("package") or header.get("category")
    kind = {"Class": "class", "Trait": "trait", "Extension": "extension"}[head.value]
    cdef = ClassDefinition(
        name=cname, kind=kind, superclass=header.get("superclass"),
        package=header.get("package"), category=header.get("category"), tag=header.get("tag"),
        instance_variables=_split_names(header.get("instVars")),
        class_variables=_split_names(header.get("classVars")),
        class_instance_variables=_split_names(header.get("classInstVars")),
        pools=_split_names(header.get("pools")),
        traits=header.get("traits"), class_traits=header.get("classTraits"),
        comment=comment, source_path=rel_path, line=line_of(source, head.start), raw_definition=header,
    )
    result.classes.append(cdef)
    result.package = str(package) if package else None
    # Methods: [ { metadata } ] Owner [class|classSide] >> pattern [ body ]
    while i < len(tokens):
        meta = {}
        try:
            if tokens[i].value == "{":
                meta, i = _ston_map(tokens, i, source)
            if i >= len(tokens):
                break
            owner_t = tokens[i]
            if owner_t.kind != "id":
                raise ParseError(f"Expected method owner, got {owner_t.value!r}", owner_t.start)
            j = i + 1
            side = "instance"
            if j < len(tokens) and tokens[j].kind == "id" and tokens[j].value in ("class", "classSide"):
                side = "class"
                j += 1
            if j >= len(tokens) or tokens[j].value != ">>":
                raise ParseError("Expected '>>' after method owner", owner_t.start)
            pattern_start = j + 1
            k = pattern_start
            while k < len(tokens) and tokens[k].value != "[":
                k += 1
            if k >= len(tokens):
                raise ParseError("Missing method body bracket", owner_t.start)
            close = _match_bracket(tokens, k, "[", "]")
            pattern_text = source[tokens[pattern_start].start:tokens[k].start].strip()
            body_text = source[tokens[k].end:tokens[close].start]
            body = _dedent_tonel_body(body_text)
            if not body.startswith("\n"):
                body = (" " + body.lstrip(" \t")) if body.strip() else ""
            method_source = pattern_text + body
            sel_hint = "".join(t.value for t in tokens[pattern_start:k] if t.kind == "keyword") or tokens[pattern_start].value
            result.methods.append(MethodSource(
                class_name=owner_t.value, side=side, selector_hint=sel_hint,
                protocol=(str(meta.get("category")) if meta.get("category") is not None else None),
                source=method_source, source_path=rel_path, line=line_of(source, tokens[pattern_start].start),
                container_format="tonel", package=result.package, is_extension=(kind == "extension"),
                in_trait=(kind == "trait"), metadata={k2: v for k2, v in meta.items() if k2 != "category"},
            ))
            i = close + 1
        except (ParseError, IndexError) as e:
            result.status = "partial"
            result.warnings.append(f"method structure: {e}")
            # resynchronise on the next metadata map or '>>' line
            i += 1
            while i < len(tokens) and tokens[i].value != "{" and not (
                    i + 1 < len(tokens) and tokens[i + 1].value == ">>"):
                i += 1
    return result


# --------------------------------------------------------------------------
# FileTree (Cypress-style directories)
# --------------------------------------------------------------------------

def _filetree_package(rel_parts: tuple[str, ...]) -> str | None:
    for p in reversed(rel_parts):
        if p.endswith(".package"):
            return p[: -len(".package")]
    return None


def read_filetree_method(source: str, rel_path: str) -> FileResult:
    p = Path(rel_path)
    result = FileResult(rel_path, "filetree_method", "ok")
    owner_dir = p.parent.parent.name
    owner = owner_dir.rsplit(".", 1)[0]
    side = "class" if p.parent.name == "class" else "instance"
    # First line is the protocol; the rest is the method source.
    nl = source.find("\n")
    protocol, method_source = (source[:nl].strip(), source[nl + 1:]) if nl >= 0 else (None, source)
    expected = p.stem
    first_sel_line = method_source.lstrip().split("\n", 1)[0]
    if protocol and not method_source.strip():
        result.status = "failed"
        result.warnings.append("FileTree method file without a method body")
        return result
    # Metadata-less variants omit the protocol line: detect by selector match.
    if protocol is not None and _pattern_selector(protocol) == expected.replace(".", ":") and _pattern_selector(first_sel_line) != expected.replace(".", ":"):
        method_source = source
        protocol = None
    result.methods.append(MethodSource(
        class_name=owner, side=side, selector_hint=expected, protocol=protocol, source=method_source.rstrip(),
        source_path=rel_path, line=2 if protocol is not None else 1, container_format="filetree",
        package=_filetree_package(p.parts), is_extension=owner_dir.endswith(".extension"),
        in_trait=owner_dir.endswith(".trait"),
    ))
    result.package = _filetree_package(p.parts)
    return result


def _pattern_selector(line: str) -> str:
    try:
        toks = tokenize(line)
    except ParseError:
        return ""
    if not toks:
        return ""
    if toks[0].kind == "keyword":
        return "".join(t.value for t in toks if t.kind == "keyword")
    return toks[0].value


def read_filetree_properties(source: str, rel_path: str) -> FileResult:
    p = Path(rel_path)
    owner_dir = p.parent.name
    result = FileResult(rel_path, "filetree_class_metadata", "metadata")
    if not owner_dir.endswith((".class", ".trait", ".extension")):
        result.format = "filetree_package_metadata"
        result.package = _filetree_package(p.parts)
        return result
    try:
        data = json.loads(source)
    except json.JSONDecodeError as e:
        result.status = "failed"
        result.warnings.append(f"invalid properties.json: {e}")
        return result
    kind = "extension" if owner_dir.endswith(".extension") else ("trait" if owner_dir.endswith(".trait") else "class")
    result.classes.append(ClassDefinition(
        name=data.get("name") or owner_dir.rsplit(".", 1)[0], kind=kind, superclass=data.get("super"),
        package=_filetree_package(p.parts), category=data.get("category"),
        instance_variables=list(data.get("instvars") or []), class_variables=list(data.get("classvars") or []),
        class_instance_variables=list(data.get("classinstvars") or []), pools=list(data.get("pools") or []),
        traits=data.get("traitComposition"), class_traits=data.get("classTraitComposition"),
        source_path=rel_path, raw_definition=data,
    ))
    result.package = _filetree_package(p.parts)
    return result


# --------------------------------------------------------------------------
# Chunk format (classic .st file-outs, Squeak/Cuis change sets)
# --------------------------------------------------------------------------

_METHODS_FOR = re.compile(r"^\s*([A-Za-z_]\w*)(\s+class)?\s+methodsFor:\s*'((?:[^']|'')*)'", re.S)
_CLASS_DEF = re.compile(
    r"^\s*([A-Za-z_]\w*)\s+(?:subclass|variableSubclass|variableByteSubclass|variableWordSubclass|weakSubclass|immediateSubclass|ephemeronSubclass):\s*#([A-Za-z_]\w*)"
    r"(?:\s+uses:\s*(?P<uses>[^\n]+?))?\s+instanceVariableNames:\s*'(?P<iv>[^']*)'"
    r"(?:\s+classVariableNames:\s*'(?P<cv>[^']*)')?"
    r"(?:\s+(?:poolDictionaries|package|category):\s*'(?P<pk>[^']*)')?",
    re.S,
)


def _split_chunks(source: str) -> list[tuple[str, int]]:
    """Split on single '!' (``!!`` is an escaped '!'). Returns (chunk, offset)."""
    chunks, buf, start, i, n = [], [], 0, 0, len(source)
    while i < n:
        c = source[i]
        if c == "!":
            if i + 1 < n and source[i + 1] == "!":
                buf.append("!")
                i += 2
                continue
            chunks.append(("".join(buf), start))
            buf, start = [], i + 1
        else:
            buf.append(c)
        i += 1
    if "".join(buf).strip():
        chunks.append(("".join(buf), start))
    return chunks


def read_chunk(source: str, rel_path: str) -> FileResult:
    result = FileResult(rel_path, "chunk", "ok")
    if rel_path.endswith(".pck.st"):
        result.dialect_hints.append("cuis_package_file")
    owner, side, protocol = None, "instance", None
    for text, offset in _split_chunks(source):
        stripped = text.strip()
        if not stripped:
            owner = None  # '! !' closes a methodsFor: section
            continue
        m = _METHODS_FOR.match(stripped)
        if m:
            owner, side, protocol = m.group(1), ("class" if m.group(2) else "instance"), m.group(3)
            continue
        if owner:
            lead = len(text) - len(text.lstrip())
            result.methods.append(MethodSource(
                class_name=owner, side=side, selector_hint=_pattern_selector(stripped.split("\n", 1)[0]),
                protocol=protocol, source=stripped, source_path=rel_path,
                line=line_of(source, offset + lead), container_format="chunk",
            ))
            continue
        cm = _CLASS_DEF.match(stripped)
        if cm:
            result.classes.append(ClassDefinition(
                name=cm.group(2), kind="class", superclass=cm.group(1), category=cm.group("pk"),
                instance_variables=(cm.group("iv") or "").split(), class_variables=(cm.group("cv") or "").split(),
                traits=(cm.group("uses") or None), source_path=rel_path, line=line_of(source, offset),
            ))
    if not result.methods and not result.classes:
        result.status = "unsupported"
        result.warnings.append("chunk file without methodsFor: sections or class definitions")
    return result


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

def classify_path(rel_path: str) -> str:
    p = Path(rel_path)
    name = p.name
    parts = p.parts
    if ".git" in parts:
        return "ignored"
    if name.endswith(TONEL_SUFFIXES) or name == "package.st":
        return "tonel"
    if name.endswith(".st") and p.parent.name in ("instance", "class") and p.parent.parent.name.endswith((".class", ".extension", ".trait")):
        return "filetree_method"
    if name == "properties.json" and any(x.endswith(".package") for x in parts):
        return "filetree_properties"
    if "monticello.meta" in parts or name in ("methodProperties.json",) or (name.endswith(".json") and any(x.endswith(".package") for x in parts)):
        return "filetree_metadata"
    if name.endswith((".st", ".cs")):
        return "smalltalk_other"
    if name.endswith((".gs", ".tpz")):
        return "gemstone_topaz"
    if name.endswith((".pcl", ".pst")):
        return "visualworks_parcel"
    return "non_source"


def read_source_file(path: Path, rel_path: str) -> FileResult | None:
    kind = classify_path(rel_path)
    if kind in ("ignored", "non_source"):
        return None
    if kind in ("gemstone_topaz", "visualworks_parcel"):
        return FileResult(rel_path, kind, "unsupported", warnings=[f"{kind} sources are not supported"], dialect_hints=[kind])
    if kind == "filetree_metadata":
        return FileResult(rel_path, "filetree_metadata", "metadata")
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            source = path.read_text(encoding="latin-1")
        except OSError as e:
            return FileResult(rel_path, kind, "failed", warnings=[f"unreadable: {e}"])
    except OSError as e:
        return FileResult(rel_path, kind, "failed", warnings=[f"unreadable: {e}"])
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    if kind == "tonel":
        return read_tonel(source, rel_path)
    if kind == "filetree_method":
        return read_filetree_method(source, rel_path)
    if kind == "filetree_properties":
        return read_filetree_properties(source, rel_path)
    # Other .st files: chunk file-outs or standalone scripts.
    if _METHODS_FOR.search(source.replace("!", "\n")) and "!" in source:
        return read_chunk(source, rel_path)
    if re.search(r"^\s*(Class|Trait|Extension)\s*\{", source, re.M) and ">>" in source:
        r = read_tonel(source, rel_path)  # Tonel content with an unusual file name
        r.format = "tonel_nonstandard_name"
        return r
    return FileResult(rel_path, "script", "unsupported",
                      warnings=["standalone script (doit); not a method container — not extracted as methods"])
