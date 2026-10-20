"""Module 7 — API and framework usage, with conservative receiver resolution.

Smalltalk dispatch is dynamic. A send is resolved to a target method ONLY with
sufficient lexical evidence:

* ``self``/``super`` sends whose selector is implemented in the defining class
  or its in-repository superclass chain (``self`` may still be overridden in a
  subclass — recorded as ``resolution_evidence``);
* sends to a global that is a class defined in the repository, when the
  selector is implemented on that class side (or its in-repo ancestors).

Literal receivers get a known receiver *class* but an external target.
Everything else is ``unknown`` — selector names alone never produce call edges.
"""
from __future__ import annotations

from collections import defaultdict
import re

LITERAL_CLASSES = {"string": "String", "symbol": "Symbol", "number": "Number", "character": "Character",
                   "array": "Array", "bytearray": "ByteArray", "boolean": "Boolean", "nil": "UndefinedObject"}


class RepositoryIndex:
    """Method/class index of one repository snapshot used for resolution."""

    def __init__(self, classes: list[dict], methods: list[dict]):
        self.superclass = {}
        for c in classes:
            if c.get("kind") != "extension" and c.get("name"):
                self.superclass.setdefault(c["name"], c.get("superclass"))
        self.defined = set(self.superclass)
        self.implementors: dict[tuple[str, str], set[str]] = defaultdict(set)  # (side, selector) -> classes
        self.methods_of: dict[tuple[str, str], set[str]] = defaultdict(set)    # (class, side) -> selectors
        for m in methods:
            if m.get("parse_status") != "ok":
                continue
            self.implementors[(m["side"], m["selector"])].add(m["class_name"])
            self.methods_of[(m["class_name"], m["side"])].add(m["selector"])

    def chain(self, cls: str) -> list[str]:
        out, seen = [], set()
        while cls and cls not in seen:
            seen.add(cls)
            out.append(cls)
            cls = self.superclass.get(cls)
        return out

    def lookup(self, cls: str, side: str, selector: str, skip_first: bool = False) -> str | None:
        for i, c in enumerate(self.chain(cls)):
            if skip_first and i == 0:
                continue
            if selector in self.methods_of.get((c, side), ()):
                return c
        return None

    def is_polymorphic(self, side: str, selector: str) -> bool:
        return len(self.implementors.get((side, selector), ())) >= 2


def resolve_send(send: dict, method: dict, index: RepositoryIndex) -> dict:
    kind, name, sel = send.get("receiver_kind"), send.get("receiver_name"), send.get("selector")
    cls, side = method["class_name"], method["side"]
    out = {"receiver_resolution": "unknown", "resolved_target": None, "resolution_evidence": None,
           "polymorphic_candidate": False}
    if kind == "self":
        owner = index.lookup(cls, side, sel)
        out["receiver_resolution"] = "self"
        if owner:
            out["resolved_target"] = f"{owner}{' class' if side == 'class' else ''}>>{sel}"
            out["resolution_evidence"] = "implemented in receiver's class chain within the repository; subclasses may override"
        else:
            out["resolution_evidence"] = "self send to a method inherited from outside the repository or not found"
        return out
    if kind == "super":
        owner = index.lookup(cls, side, sel, skip_first=True)
        out["receiver_resolution"] = "super"
        if owner:
            out["resolved_target"] = f"{owner}{' class' if side == 'class' else ''}>>{sel}"
            out["resolution_evidence"] = "static super lookup within repository superclass chain"
        else:
            out["resolution_evidence"] = "super implementation outside the repository"
        return out
    if kind == "global" and name:
        out["receiver_resolution"] = "global_class" if name[:1].isupper() else "global"
        if name in index.defined:
            owner = index.lookup(name, "class", sel)
            if owner:
                out["resolved_target"] = f"{owner} class>>{sel}"
                out["resolution_evidence"] = "class-side method of a repository class"
            else:
                out["resolution_evidence"] = f"receiver is repository class {name}; selector inherited from outside"
        else:
            out["resolution_evidence"] = f"receiver is external global {name}; target outside the repository"
        return out
    if kind == "literal":
        out["receiver_resolution"] = "literal"
        out["resolution_evidence"] = f"receiver class {LITERAL_CLASSES.get(name, name)} (target outside the repository)"
        return out
    if kind == "block":
        out["receiver_resolution"] = "literal"
        out["resolution_evidence"] = "receiver class FullBlockClosure"
        return out
    if kind == "thisContext":
        out["receiver_resolution"] = "thisContext"
        out["resolution_evidence"] = "receiver class Context"
        return out
    out["polymorphic_candidate"] = index.is_polymorphic("instance", sel)
    return out


class FrameworkMatcher:
    def __init__(self, spec: dict):
        self.rules = []
        for fw in (spec or {}).get("frameworks", []):
            exact = set(fw.get("exact") or [])
            prefixes = [re.compile(p) for p in fw.get("prefixes") or []]
            self.rules.append((fw["name"], fw.get("kind", "framework"), exact, prefixes))
        self._cache: dict[str, tuple[str | None, str | None]] = {}

    def match(self, name: str) -> tuple[str | None, str | None]:
        if name in self._cache:
            return self._cache[name]
        res = (None, None)
        for fw, kind, exact, prefixes in self.rules:
            if name in exact or any(p.search(name) for p in prefixes):
                res = (fw, kind)
                break
        self._cache[name] = res
        return res


def api_references(method: dict, index: RepositoryIndex, matcher: FrameworkMatcher) -> list[dict]:
    counts = defaultdict(int)
    for g in method.get("referenced_globals") or []:
        counts[g["name"]] += 1
    out = []
    for name, n in counts.items():
        fw, kind = matcher.match(name)
        defined = name in index.defined
        out.append({"name": name, "kind": "class" if name in index.defined or name[:1].isupper() else "global",
                    "defined_in_repository": defined, "framework": "(repository)" if defined and not fw else fw,
                    "framework_kind": kind, "occurrences": n})
    return out
