"""Method filters (phase 3): what happens to each method extracted from the selected repositories.

Every rule is decided on the method's **abstract syntax tree** (the parser's AST: Method › Sequence of statements
› Return / Assignment / Message / Cascade / Variable / Literal / Block / DynamicArray, with Pragma nodes) and on the
**definition of its class** (instance, class and class-instance variables), never on the text alone. Variables are
resolved like the compiler does: method arguments and temporaries, then the variables of the class (instance-side
or class-side), then pseudo-variables (self, super, thisContext, true, false, nil), then globals.

One rule per kind of method; a method keeps every reason that applies; its decision is the strongest:

    exclude  >  keep_test  >  review  >  keep

| kind of method (AST pattern) | decision | why |
|---|---|---|
| duplicate: same code as an older method (identical, or once comments and layout are removed) | exclude | the same code must not be counted twice |
| in a mirror repository (mostly copies of an older one) | exclude | repeated Pharo core or forked code |
| cannot be parsed (no AST) | exclude | its structure cannot be analysed reliably |
| simple getter: the only statement is ``^ v``, v a variable of the class | exclude | no behaviour to evaluate |
| simple setter: one argument a; statements ``v := a`` (v a variable of the class), optionally ``^ self`` / ``^ a`` | exclude | no behaviour to evaluate |
| returns a constant: the only statement is ``^ literal`` (number, string, symbol, character, literal array, true / false / nil, a brace array of literals) | exclude | limited behaviour |
| empty: no statement | exclude | no implementation |
| placeholder: the only statement is ``self subclassResponsibility`` / ``shouldBeImplemented`` / ``notYetImplemented`` / ``shouldNotImplement`` / ``explicitRequirement`` / ``halt``, ``^ self``, or ``self error: '... not implemented ...'`` | exclude | no implementation |
| generated boilerplate: a method of a package manifest or of a Metacello baseline / configuration class; a ``<generated>`` pragma; embedded data (literals are > 80% of a long method) | exclude | repetitive, tool-generated |
| trivial initialisation: an ``initialize…`` whose statements are only ``super initialize`` and ``v := literal`` / ``v := SomeClass new`` | exclude | trivial assignments |
| simple message forwarding: one statement, ``[^] receiver selector args`` where the receiver is a variable of the class, self or an argument, and every argument is an argument of the method; no block | review | forwarding sometimes implements behaviour |
| non-trivial initialisation: any other ``initialize…`` or a lazy initialiser (``^ v ifNil: [ v := … ]``) | review | set-up logic, kept but marked |
| very long method: more than ``long_method_lines`` lines or ``long_method_statements`` statements (all levels) | review | valuable but needs more context |
| requires dependencies: references global names neither its project nor a standard Pharo 13 image defines | review | kept when the dependency is manageable |
| test method (a test class) | keep_test | kept separately (test generation; never mixed in production frequencies) |

``review`` methods are kept in the samples, marked (``method_filters.include_review``), with their reasons.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

PLACEHOLDER_SELECTORS = {"subclassResponsibility", "shouldBeImplemented", "notYetImplemented", "shouldNotImplement",
                         "explicitRequirement", "halt", "halt:", "flag:"}
PSEUDO = {"self", "super", "thisContext", "true", "false", "nil"}
CONSTANT_PSEUDO = {"true", "false", "nil"}
GENERATED_CLASS_RE = re.compile(r"^(?:Manifest\w*|ConfigurationOf\w+|BaselineOf\w+)$")
GENERATED_SUPER = {"PackageManifest", "BaselineOf", "ConfigurationOf"}
NOT_IMPLEMENTED_RE = re.compile(r"not\s+(?:yet\s+)?implemented|to\s*do|todo|implement me", re.I)


# ---------------------------------------------------------------------------------------------- AST helpers
def walk(node):
    """Every node of an AST (depth first)."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


def statements(ast: dict) -> list[dict]:
    body = (ast or {}).get("body") or {}
    return list(body.get("statements") or [])


def unwrap_return(st: dict) -> tuple[bool, dict]:
    """(is a return, the returned / evaluated expression)."""
    if st.get("type") == "Return":
        return True, st.get("value") or {}
    return False, st


def is_constant(expr: dict) -> bool:
    t = expr.get("type")
    if t == "Literal":
        return True
    if t == "Variable" and expr.get("name") in CONSTANT_PSEUDO:
        return True
    if t == "DynamicArray":                       # { 1. 2. 3 } of constants only
        return all(is_constant(s) for s in expr.get("statements") or [])
    return False


def is_var(expr, names: set) -> bool:
    """A Variable node (or, as the parser stores assignment targets, a bare name) whose name is in ``names``."""
    if isinstance(expr, str):
        return expr in names
    return isinstance(expr, dict) and expr.get("type") == "Variable" and expr.get("name") in names


def is_self_send(expr: dict, selectors: set) -> bool:
    return (expr.get("type") == "Message" and is_var(expr.get("receiver") or {}, {"self"})
            and expr.get("selector") in selectors)


class Scope:
    """The names a method can see: arguments, temporaries (method and blocks), the variables of its class."""

    def __init__(self, ast: dict, class_vars: set):
        self.args = set((ast or {}).get("arguments") or [])
        temps = set()
        for n in walk(ast):
            if n.get("type") == "Sequence":
                temps |= {t if isinstance(t, str) else t.get("name") for t in n.get("temporaries") or []}
            if n.get("type") == "Block":
                temps |= {a if isinstance(a, str) else a.get("name") for a in n.get("arguments") or []}
        self.temps = temps
        self.class_vars = class_vars

    def globals(self, ast: dict) -> set:
        """Names that resolve to globals: neither local, nor of the class, nor pseudo-variables."""
        out = set()
        for n in walk(ast):
            if n.get("type") == "Variable":
                nm = n.get("name") or ""
                if nm and nm not in self.args and nm not in self.temps and nm not in self.class_vars and nm not in PSEUDO:
                    out.add(nm)
        return out


def analyse(ast: dict, class_vars: set, selector: str, source: str, pragmas: list) -> dict:
    """AST rules of one method: exclusion and review reasons, statement count (all levels), globals used."""
    ex, rev = [], []
    if not isinstance(ast, dict) or ast.get("type") != "Method":
        return {"exclude": ["cannot be parsed (no AST)"], "review": [], "n_statements_all": 0, "globals": []}
    scope = Scope(ast, class_vars)
    sts = statements(ast)
    n_all = sum(1 for n in walk(ast) if n.get("type") == "Sequence" for _ in (n.get("statements") or []))
    prag = {p.get("selector") for p in (ast.get("pragmas") or []) if isinstance(p, dict)} | set(pragmas or [])
    if prag & {"generated", "gtGenerated"}:
        ex.append("generated boilerplate (<generated> pragma)")
    if not sts:
        ex.append("empty method")
    elif len(sts) == 1:
        ret, e = unwrap_return(sts[0])
        if ret and is_var(e, class_vars):
            ex.append("simple getter")
        elif ret and is_constant(e):
            ex.append("returns a constant")
        elif (is_self_send(e, PLACEHOLDER_SELECTORS) or (ret and is_var(e, {"self"}))
              or (is_self_send(e, {"error:"}) and any(a.get("type") == "Literal" and NOT_IMPLEMENTED_RE.search(str(a.get("value")))
                                                      for a in e.get("arguments") or []))):
            ex.append("placeholder")
    if not ex and len(scope.args) == 1 and 1 <= len(sts) <= 2:
        (arg,) = scope.args
        first = sts[0]
        if (first.get("type") == "Assignment" and is_var(first.get("variable") or {}, class_vars)
                and is_var(first.get("value") or {}, {arg})):
            rest_ok = len(sts) == 1 or (unwrap_return(sts[1])[0] and is_var(unwrap_return(sts[1])[1], {"self", arg}))
            if rest_ok:
                ex.append("simple setter")
    if not ex and (selector or "").startswith("initialize") and sts:
        def trivial(st):
            if st.get("type") == "Message" and is_var(st.get("receiver") or {}, {"super"}) and st.get("selector") == "initialize":
                return True
            if st.get("type") == "Assignment" and is_var(st.get("variable") or {}, class_vars):
                v = st.get("value") or {}
                if is_constant(v):
                    return True
                if (v.get("type") == "Message" and (v.get("receiver") or {}).get("type") == "Variable"
                        and str((v.get("receiver") or {}).get("name", ""))[:1].isupper()
                        and v.get("selector") in ("new", "new:") and all(is_constant(a) for a in v.get("arguments") or [])):
                    return True
            return False
        if all(trivial(st) for st in sts):
            ex.append("trivial initialisation")
    if not ex and len(sts) == 1:
        _, e = unwrap_return(sts[0])
        if (e.get("type") == "Message" and is_var(e.get("receiver") or {}, class_vars | {"self"} | scope.args)
                and all(is_var(a, scope.args) for a in e.get("arguments") or [])
                and not any(n.get("type") == "Block" for n in walk(e))):
            rev.append("simple message forwarding")
    if not ex:
        lazy = (len(sts) == 1 and unwrap_return(sts[0])[0]
                and (unwrap_return(sts[0])[1].get("selector") in ("ifNil:", "ifNil:ifNotNil:"))
                and is_var((unwrap_return(sts[0])[1].get("receiver") or {}), class_vars))
        if (selector or "").startswith("initialize") or lazy:
            rev.append("non-trivial initialisation")
    lit_chars = sum(len(str(n.get("value") or "")) for n in walk(ast) if n.get("type") == "Literal")
    if not ex and len(source or "") > 800 and lit_chars / max(len(source), 1) > 0.8:
        ex.append("generated boilerplate (embedded data literal)")
    return {"exclude": ex, "review": rev, "n_statements_all": n_all, "globals": sorted(scope.globals(ast))}


def class_variables(cls: dict, side: str) -> set:
    """Variables a method of this class sees on its side (instance side: instance + class variables; class side:
    class-instance + class variables)."""
    def names(xs):
        return {x if isinstance(x, str) else (x or {}).get("name") for x in (xs or [])} - {None}
    cv = names(cls.get("class_variables"))
    if side == "class":
        return names(cls.get("class_instance_variables")) | cv
    return names(cls.get("instance_variables")) | cv


# ---------------------------------------------------------------------------------------------- decisions
def classify(df: pd.DataFrame, cfg: dict, mirrors: set, image_globals: set, repo_classes: dict) -> pd.DataFrame:
    """decision and reasons for every method, from the AST facts computed at extraction (columns ast_exclude,
    ast_review, ast_globals, n_statements_all) and the repository-level facts (copies, mirrors, test classes)."""
    mf = cfg["method_filters"]
    long_lines = int(mf.get("long_method_lines", 60))
    long_stmts = int(mf.get("long_method_statements", 40))
    decisions, reasons = [], []
    for r in df.itertuples(index=False):
        ex, rev = [], []
        if isinstance(r.duplicate_of, str) and r.duplicate_of:
            ex.append(f"duplicate ({r.duplicate_kind}) of {r.duplicate_of}")
        if r.repository in mirrors:
            ex.append("mirror repository")
        if r.parse_status != "ok":
            ex.append("cannot be parsed")
        if GENERATED_CLASS_RE.match(r.class_name or "") or (r.superclass in GENERATED_SUPER):
            ex.append("generated boilerplate (manifest / Metacello baseline or configuration)")
        ex += [x for x in (r.ast_exclude or "").split("; ") if x]
        if not ex:
            rev += [x for x in (r.ast_review or "").split("; ") if x]
            if int(r.n_lines or 0) > long_lines or int(r.n_statements_all or 0) > long_stmts:
                rev.append(f"very long method (> {long_lines} lines or > {long_stmts} statements)")
            if image_globals:
                own = repo_classes.get(r.repository, set())
                ext = sorted(g for g in (r.ast_globals if r.ast_globals is not None else [])
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
        reasons.append("; ".join(dict.fromkeys(ex + rev + (["test method (kept separately)"] if r.is_test and not ex else []))))
    return pd.DataFrame({"method_id": df["method_id"].values, "repository": df["repository"].values,
                         "is_test": df["is_test"].values, "decision": decisions, "reasons": reasons})


def first_rule(reasons: str) -> str:
    """The rule a reason string starts with, for counting (``duplicate (exact) of M…`` → ``duplicate``)."""
    if not reasons:
        return "kept"
    r = reasons.split(";")[0].strip()
    return re.sub(r"\s*[(:].*$", "", r)
