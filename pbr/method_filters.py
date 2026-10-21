"""Method filters (phase 3): what happens to each method extracted from the selected repositories.

One rule per kind of method; each rule gives a decision and a reason, and every method keeps all the reasons that
apply (``method_filters.parquet``). The decision of a method is the strongest of its rules:

    exclude  >  keep_test  >  review  >  keep

| kind of method | decision | why |
|---|---|---|
| duplicate (non-canonical copy, exact or once comments/layout are removed) | exclude | the same code must not be counted twice |
| in a mirror repository (mostly copies of an older one) | exclude | repeated Pharo core or forked code |
| cannot be parsed | exclude | its structure cannot be analysed reliably |
| simple getter (only returns an instance variable) | exclude | no behaviour to evaluate |
| simple setter (only assigns an instance variable) | exclude | no behaviour to evaluate |
| returns a constant (``^ 42``, ``^ 'name'``, ``^ #symbol``, ``^ true``...) | exclude | limited behaviour |
| empty or placeholder (no statement, ``self subclassResponsibility``, ``self shouldBeImplemented``, ...) | exclude | no implementation |
| generated boilerplate (package manifests, Metacello baselines / configurations, embedded data literals) | exclude | repetitive, tool-generated |
| trivial initialisation (``initialize`` that only calls super and assigns literals / ``new``) | exclude | trivial assignments |
| simple message forwarding (one statement, one message send, no block) | review | forwarding sometimes implements behaviour |
| non-trivial initialisation | review | set-up logic, kept but marked |
| very long method (``long_method_lines``) | review | valuable but needs more context |
| requires dependencies (references classes neither its project nor a standard Pharo image defines) | review | kept when the dependency is manageable |
| test method | keep_test | kept separately (test generation; never mixed in production frequencies) |

``review`` methods are kept in the samples, marked (``method_filters.include_review``); the reasons go with them.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

PLACEHOLDER_SENDS = ("subclassResponsibility", "shouldBeImplemented", "notYetImplemented", "shouldNotImplement",
                     "explicitRequirement", "halt", "flag:")
LITERAL = r"(?:-?\d[\w.]*|'(?:[^']|'')*'|#\w[\w:]*|#'(?:[^']|'')*'|#\(.*\)|#\[.*\]|\$.|true|false|nil|\{\s*\})"
CONST_RE = re.compile(rf"^\^\s*{LITERAL}\s*\.?$", re.S)
PLACEHOLDER_RE = re.compile(rf"^(?:\^\s*)?self\s+(?:{'|'.join(re.escape(s) for s in PLACEHOLDER_SENDS)})(?:\s+'(?:[^']|'')*')?\s*\.?$"
                            r"|^\^\s*self\s*\.?$|^self\s+error:\s*'[^']*(?:not\s+(?:yet\s+)?implemented|todo|to do)[^']*'\s*\.?$", re.I | re.S)
INIT_STMT_RE = re.compile(rf"^(?:super\s+initialize|\w+\s*:=\s*(?:{LITERAL}|[A-Z]\w*\s+new(?::\s*\S+)?|[A-Z]\w*\s+new\s+\w+))$", re.S)
GENERATED_CLASS_RE = re.compile(r"^(?:Manifest\w*|ConfigurationOf\w+|BaselineOf\w+)$")
GENERATED_SUPER = {"PackageManifest", "BaselineOf", "ConfigurationOf"}


def body_statements(norm_source: str) -> list[str]:
    """The statements of a method body, from its comment-free, whitespace-collapsed source: the header (message
    pattern) and the temporaries declaration are removed; statements are split on top-level periods."""
    s = norm_source.strip()
    # header: keyword pattern (kw: arg)+, binary (op arg) or unary (word)
    m = re.match(r"^(?:(?:[a-zA-Z_]\w*:\s*[a-zA-Z_]\w*\s*)+|[-+*/\\<>=~@%|&?,]+\s*[a-zA-Z_]\w*\s*|[a-zA-Z_]\w*\s*)", s)
    s = s[m.end():] if m else s
    s = re.sub(r"^<[^>]*>\s*", "", s)                              # pragmas
    s = re.sub(r"^\|[^|]*\|\s*", "", s)                            # temporaries
    s = re.sub(r"^(?:<[^>]*>\s*)+", "", s)
    out, depth, cur, quote = [], 0, [], False
    for ch in s:
        if ch == "'":
            quote = not quote
        if not quote:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "." and depth == 0:
                if "".join(cur).strip():
                    out.append("".join(cur).strip())
                cur = []
                continue
        cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def literal_share(src: str) -> float:
    """Share of a method's characters inside string / byte-array / literal-array literals (embedded data)."""
    lit = sum(len(x) for x in re.findall(r"'(?:[^']|'')*'|#\[[^\]]*\]|#\([^)]*\)", src))
    return lit / max(len(src), 1)


def classify(df: pd.DataFrame, cfg: dict, mirrors: set, image_globals: set, repo_classes: dict, norm) -> pd.DataFrame:
    """decision and reasons for every method. df: method rows (methods.parquet columns); norm: the function that
    removes comments and collapses whitespace (corpus.strip_comments)."""
    mf = cfg["method_filters"]
    long_lines = int(mf.get("long_method_lines", 60))
    decisions, reasons = [], []
    for r in df.itertuples(index=False):
        ex, rev = [], []
        if r.duplicate_of is not None and not (isinstance(r.duplicate_of, float) and np.isnan(r.duplicate_of)):
            ex.append(f"duplicate ({r.duplicate_kind}) of {r.duplicate_of}")
        if r.repository in mirrors:
            ex.append("mirror repository")
        if r.parse_status != "ok":
            ex.append("cannot be parsed")
        stmts = None
        if not ex:
            if r.is_getter:
                ex.append("simple getter")
            elif r.is_setter:
                ex.append("simple setter")
            if GENERATED_CLASS_RE.match(r.class_name or "") or (r.superclass in GENERATED_SUPER):
                ex.append("generated boilerplate (manifest / Metacello baseline or configuration)")
            elif r.source_chars > 800 and literal_share(r.source) > 0.8:
                ex.append("generated boilerplate (embedded data literal)")
            elif any(p in ("generated", "gtGenerated") for p in (r.pragmas if r.pragmas is not None else [])):
                ex.append("generated boilerplate (<generated> pragma)")
        if not ex:
            stmts = body_statements(norm(r.source or ""))
            if not stmts or r.is_empty:
                ex.append("empty method")
            elif len(stmts) == 1 and PLACEHOLDER_RE.match(stmts[0]):
                ex.append("placeholder (" + stmts[0][:40] + ")")
            elif len(stmts) == 1 and CONST_RE.match(stmts[0]):
                ex.append("returns a constant")
            elif (r.selector or "").startswith("initialize") and all(INIT_STMT_RE.match(s_) for s_ in stmts):
                ex.append("trivial initialisation")
        if not ex:
            if len(stmts) == 1 and int(r.n_message_sends or 0) == 1 and int(r.n_blocks or 0) == 0 and not r.is_test:
                rev.append("simple message forwarding")
            if (r.selector or "").startswith("initialize") or r.is_lazy_initializer:
                rev.append("non-trivial initialisation")
            if int(r.n_lines or 0) > long_lines or r.over_long:
                rev.append(f"very long method (> {long_lines} lines)")
            if image_globals:
                own = repo_classes.get(r.repository, set())
                ext = sorted(g for g in (r.referenced_globals if r.referenced_globals is not None else [])
                             if g and g[0].isupper() and g not in own and g not in image_globals)
                if ext:
                    rev.append("requires dependencies: " + ", ".join(ext[:4]) + (" ..." if len(ext) > 4 else ""))
        if ex:
            d = "exclude"
        elif r.is_test:
            d = "keep_test"
        elif rev:
            d = "review"
        else:
            d = "keep"
        decisions.append(d)
        reasons.append("; ".join(ex + rev + (["test method (kept separately)"] if r.is_test and not ex else [])))
    return pd.DataFrame({"method_id": df["method_id"].values, "repository": df["repository"].values,
                         "is_test": df["is_test"].values, "decision": decisions, "reasons": reasons})


def first_rule(reasons: str) -> str:
    """The rule a reason string starts with, for counting (``duplicate (exact) of M…`` → ``duplicate``)."""
    if not reasons:
        return "kept"
    r = reasons.split(";")[0].strip()
    return re.sub(r"\s*[(:].*$", "", r)
