"""Open coding of methods: the prompt, the parsing of the answer, and the codebook.

Each method gets a one-sentence *summary* and a *task*: a short, general name for the kind of
programming task the method performs (not its application domain). The summary comes first in
the answer so the task is named after the model has said what the code does.

Every method is coded independently: the prompt does not list the categories found so far.
Small models given such a list copy it even when no entry fits (measured on the corpus: a
listed prompt put a visitor under "Object initialization" and a snapshot writer under "Numeric
computation"; without the list the same model answered "Visitor" and "Serialization").
Convergence comes from the codebook instead, which merges a new name into an existing one when
they are (1) equal after normalisation, (2) spelling variants (character similarity), or (3) the
same meaning: the cosine similarity of their embeddings by a small sentence-embedding model is
above ``semantic_threshold`` ("Visitor implementation" -> "Visitor pattern", "Calculation" ->
"Computation"). Without (3), open codes from a small model never saturate (measured on the corpus:
102 categories and 272 single-use names after 1,384 methods). Every merge is recorded with its
similarity, so experts can undo it.

A name becomes a category at the end of a round once at least ``min_support`` methods and at least
``min_share`` of all methods labelled so far have it; until then it is a candidate, listed for the
experts but not in the taxonomy. The share matters: a small model gives a one-off name to about
40% of methods, so "seen twice" never saturates (measured: 105 categories after 1,384 methods and
still growing), while "seen in 0.5% of the methods" does. It also says what saturation means: a
task with prevalence p goes unseen in n methods with probability (1-p)^n (0.1% for p = 0.5%,
n = 1,384). The few-shot
examples carry ten seed categories; they are reported as such.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import re

import numpy as np

PROMPT_VERSION = "task-coding-v2"
STOP = ["\n\n", "\n### Method"]

HEADER = ("Each Pharo method below is labelled with the programming task it performs.\n"
          "Summary: one sentence saying what the method does.\n"
          "Task: a general category of 1 to 4 words naming the kind of programming task (not the application domain).\n\n")

# (class, source, task, summary): the seed categories of the codebook
EXAMPLES = [
    ("Invoice", "date: aDate\n\tdate := aDate", "Accessor", "Sets the date of the invoice."),
    ("BaselineOfShop", "baseline: spec\n\t<baseline>\n\tspec for: #common do: [\n\t\tspec package: 'Shop-Core'.\n"
     "\t\tspec package: 'Shop-Tests' with: [ spec requires: #('Shop-Core') ] ]",
     "Project configuration", "Declares the packages of the Shop project and their dependencies."),
    ("TodoListPresenter", "initializePresenters\n\tlist := self newList.\n\taddButton := self newButton label: 'Add'; yourself",
     "User interface", "Creates the list and the Add button of the todo list window."),
    ("Matrix", "* aMatrix\n\t| result |\n\tresult := Matrix rows: self rowCount columns: aMatrix columnCount.\n"
     "\t1 to: self rowCount do: [ :i |\n\t\t1 to: aMatrix columnCount do: [ :j |\n"
     "\t\t\tresult at: i at: j put: ((self row: i) dot: (aMatrix column: j)) ] ].\n\t^ result",
     "Numeric computation", "Multiplies the receiver by another matrix."),
    ("ParserTest", "setUp\n\tsuper setUp.\n\tparser := JSONParser new",
     "Test fixture", "Creates the parser used by every test of the class."),
    ("ParserTest", "testParseEmptyArray\n\tself assert: (parser parse: '[]') isEmpty",
     "Unit test", "Checks that parsing an empty JSON array answers an empty collection."),
    ("Point", "x\n\t^ x", "Accessor", "Returns the x coordinate of the point."),
    ("OrderedCollectionTest", "testAddFirst\n\t| c |\n\tc := OrderedCollection new.\n\tc addFirst: 3.\n"
     "\tself assert: c first equals: 3", "Unit test", "Checks that addFirst: puts the element at the front of the collection."),
    ("Invoice", "initialize\n\tsuper initialize.\n\tlines := OrderedCollection new.\n\tdate := Date today",
     "Object initialization", "Sets up an empty list of lines and today's date for a new invoice."),
    ("Library", "overdueBooksOn: aDate\n\t^ books select: [ :each | each dueDate < aDate ]",
     "Collection filtering", "Answers the books whose due date is before the given date."),
    ("Money", "printOn: aStream\n\taStream nextPutAll: amount printString; space; nextPutAll: currency",
     "Printing", "Writes the amount followed by the currency on a stream."),
    ("ConfigLoader", "loadFrom: aFileReference\n\t^ [ STON fromString: aFileReference contents ]\n\t\ton: Error\n"
     "\t\tdo: [ :e | self defaultConfiguration ]",
     "Deserialization", "Reads a configuration from a STON file, falling back to the default on error."),
]
SEED_CATEGORIES = list(dict.fromkeys(e[2] for e in EXAMPLES))


def _block(cls: str, source: str, task: str | None = None, summary: str | None = None) -> str:
    s = f"### Method of {cls}\n{source.strip()}\n### Summary:"
    if task:
        s += f" {summary}\n### Task: {task}\n\n"
    return s


def method_source(method: dict, max_lines: int) -> str:
    lines = (method.get("source") or "").strip().splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["\t\"...\""]
    return "\n".join(lines)


def build_prompt(method: dict, max_lines: int = 40) -> str:
    """Few-shot completion prompt; the answer starts after "### Summary:"."""
    p = HEADER
    for cls, src, task, summary in EXAMPLES:
        p += _block(cls, src, task, summary)
    cls = method["class_name"] + (" class" if method.get("side") == "class" else "")
    return p + _block(cls, method_source(method, max_lines))


_TASK = re.compile(r"\n?#*\s*Task\s*:\s*(.*)", re.IGNORECASE)


def parse_answer(text: str) -> tuple[str, str | None]:
    """(summary, task name or None) from a completion that starts after "### Summary:"."""
    text = (text or "").strip()
    m = _TASK.search(text)
    if not m:
        return text.splitlines()[0].strip() if text else "", None
    summary = text[:m.start()].strip().splitlines()[0].strip() if text[:m.start()].strip() else ""
    return summary, clean_name(m.group(1).splitlines()[0] if m.group(1) else "")


_CODE = re.compile(r"[:=^\[\]{}|#<>]|\.\w|\w\.\w")


def clean_name(raw: str) -> str | None:
    """A usable category name, or None when the answer is not one (code, a sentence, empty)."""
    s = raw.strip().strip("\"'`*.,;").strip()
    s = re.sub(r"\s+", " ", s)
    if not s or _CODE.search(s) or len(s.split()) > 5 or len(s) > 48 or not re.match(r"[A-Za-z]", s):
        return None
    return s[0].upper() + s[1:]


def key(name: str) -> str:
    """Normalised form: case, punctuation, plural and a trailing generic word do not count."""
    words = re.findall(r"[a-z0-9]+", name.lower())
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words]
    while len(words) > 1 and words[-1] in ("task", "method", "operation", "logic", "handling"):
        words = words[:-1]
    return " ".join(words)


def opposite(a: str, b: str) -> bool:
    """Names that differ by a negating prefix ("Serialization" / "Deserialization"): spelling and
    embeddings put them close together, but they are opposite tasks and must never be merged."""
    ka, kb = key(a).replace(" ", ""), key(b).replace(" ", "")
    return any(ka == p + kb or kb == p + ka for p in ("de", "un", "dis", "in", "non"))


class Codebook:
    """Task categories, their support and the round in which they entered the taxonomy.

    A name enters the taxonomy when ``admit`` finds it has ``min_support`` methods and
    ``min_share`` of the labelled methods (seed categories too: a seed nobody uses is not a
    category of this corpus). Its ``admitted_round`` is what the saturation loop counts."""

    def __init__(self, min_support: int = 2, similarity: float = 0.88, seeds: list[str] = SEED_CATEGORIES,
                 embed=None, semantic_threshold: float = 0.78, min_share: float = 0.0):
        self.min_support = max(1, int(min_support))
        self.min_share = float(min_share)
        self.similarity = float(similarity)
        self.embed = embed                 # list[str] -> unit vectors, or None (lexical merging only)
        self.semantic_threshold = float(semantic_threshold)
        self.vectors: dict[str, np.ndarray] = {}
        self._matrix, self._matrix_names = None, None
        self.cats: dict[str, dict] = {}   # canonical name -> info
        self.alias: dict[str, str] = {}   # key -> canonical name
        for s in seeds:
            self._new(s, None, seed=True)

    def _new(self, name: str, round_no: int | None, seed: bool = False) -> dict:
        self.cats[name] = {"name": name, "seed": seed, "support": 0, "admitted_round": None,
                           "first_round": round_no, "variants": set(), "merged": {}}
        self.alias[key(name)] = name
        return self.cats[name]

    def match(self, name: str) -> str | None:
        """Lexical match only: normalised form, then spelling variants."""
        k = key(name)
        if k in self.alias:
            return self.alias[k]
        best, score = None, 0.0
        for canon in self.cats:
            if opposite(name, canon):
                continue
            r = SequenceMatcher(None, k, key(canon)).ratio()
            if r > score:
                best, score = canon, r
        return best if score >= self.similarity else None

    def _vector(self, name: str):
        if name not in self.vectors:
            self.vectors[name] = np.asarray(self.embed([name])[0], dtype=np.float32)
        return self.vectors[name]

    def semantic_match(self, name: str) -> tuple[str | None, float]:
        """The existing category closest in meaning, when it is close enough."""
        if self.embed is None or not self.cats:
            return None, 0.0
        names = list(self.cats)
        if self._matrix is None or self._matrix_names != names:
            self._matrix = np.stack([self._vector(n) for n in names])
            self._matrix_names = names
        sims = self._matrix @ self._vector(name)
        for i, n in enumerate(names):
            if opposite(name, n):
                sims[i] = -1.0
        i = int(np.argmax(sims))
        score = float(sims[i])
        return (names[i], score) if score >= self.semantic_threshold else (None, score)

    def observe(self, name: str | None, round_no: int) -> str | None:
        """Record one method labelled ``name``; returns its canonical name."""
        if not name:
            return None
        canon = self.match(name)
        if canon is None:
            canon, score = self.semantic_match(name)
            if canon is not None:
                self.cats[canon]["merged"][name] = round(score, 3)
        c = self.cats[canon] if canon else self._new(name, round_no)
        canon = c["name"]
        if c["first_round"] is None:
            c["first_round"] = round_no
        c["support"] += 1
        c["variants"].add(name)
        self.alias.setdefault(key(name), canon)
        return canon

    def admit(self, round_no: int, labelled: int) -> list[str]:
        """End of a round: the names that now reach both thresholds enter the taxonomy."""
        need = max(self.min_support, self.min_share * labelled)
        new = [n for n, c in self.cats.items() if c["admitted_round"] is None and c["support"] >= need]
        for n in new:
            self.cats[n]["admitted_round"] = round_no
        return new

    def in_taxonomy(self, canon: str, round_no: int | None = None) -> bool:
        r = self.cats[canon]["admitted_round"]
        return r is not None and (round_no is None or r <= round_no)

    def taxonomy(self, round_no: int | None = None) -> list[str]:
        return [n for n in self.cats if self.in_taxonomy(n, round_no)]

    def final(self, name: str | None) -> str | None:
        """The category a raw name ends up in, or None when it never reached ``min_support``."""
        if not name:
            return None
        canon = self.match(name) or self.semantic_match(name)[0]
        return canon if canon and self.in_taxonomy(canon) else None
