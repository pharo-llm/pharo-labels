"""Model calls of the benchmark generation: write a task from a mined seed method, repair it
from the image's error report, add tests against surviving mutants, write the task statement,
and (a different model) solve the task from the statement alone.

Models only propose. Whether a proposal is kept is decided by the gates (gates.py) from what
the Pharo image observed. Every response is cached by model and prompt, so a run resumes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

from ..extraction.smalltalk_parser import ParseError, parse_method

PROMPT_VERSION = "benchgen-1"

_METHOD = {"type": "object", "properties": {"side": {"type": "string", "enum": ["instance", "class"]},
                                             "source": {"type": "string"}},
           "required": ["side", "source"], "additionalProperties": False}
_TEST = {"type": "object", "properties": {"source": {"type": "string"}}, "required": ["source"],
         "additionalProperties": False}
_NAMES = {"type": "array", "items": {"type": "string"}}

TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "class_name": {"type": "string"}, "superclass": {"type": "string"},
        "instance_variables": _NAMES, "class_comment": {"type": "string"},
        "context_methods": {"type": "array", "items": _METHOD},
        "target_methods": {"type": "array", "items": _METHOD},
        "test_class_name": {"type": "string"}, "test_instance_variables": _NAMES,
        "tests": {"type": "array", "items": _TEST},
    },
    "required": ["title", "class_name", "superclass", "instance_variables", "class_comment", "context_methods",
                 "target_methods", "test_class_name", "test_instance_variables", "tests"],
    "additionalProperties": False,
}
TESTS_SCHEMA = {"type": "object", "properties": {"tests": {"type": "array", "items": _TEST}}, "required": ["tests"],
                "additionalProperties": False}
DESCRIPTION_SCHEMA = {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"],
                      "additionalProperties": False}
SOLUTION_SCHEMA = {"type": "object", "properties": {"methods": {"type": "array", "items": _METHOD}}, "required": ["methods"],
                   "additionalProperties": False}

AUTHOR_SYSTEM = """You write programming tasks for a Pharo (Smalltalk) code-generation benchmark.
Each task is one small class, one or two methods that a model will have to implement (the
target methods), and SUnit tests that decide whether an implementation is correct.

Your output is loaded into a clean Pharo {pharo} image and executed, and it is discarded if
anything fails, so follow these rules exactly:
- Self-contained: use only classes and messages that exist in a standard Pharo {pharo} image.
  No external library, no file, network, clock (now, today), user interface or random value.
- The class name starts with "{prefix}" and the test class name is the class name followed by
  "Test". Do not reuse the names of the seed method's project.
- "source" is the full source of ONE method: the message pattern on the first line, then the
  body. No class name, no ">>", no Tonel brackets, no "!" chunk marks.
- context_methods are given to the solver as they are (initialize, accessors, helpers).
  target_methods are your reference solution; the solver sees only their message pattern.
  Each target method must contain real logic (at least a few statements, with a condition,
  a loop or a collection operation); a getter or a one-message wrapper is not a task.
- The target methods must perform the programming operation named in the request, using the
  idiomatic Pharo messages for it, as the seed method does.
- Tests: {min_tests} to 8 test methods named test…, plus setUp if useful. Each assertion compares
  the result with a literal expected value (self assert: … equals: …). Cover the normal
  case, boundaries (empty, one element, limits) and each branch of the logic. Tests call only
  the context and target methods. Tests must be deterministic and independent of each other.
- Work out every expected value by hand, carefully: a wrong expected value rejects the task.
Answer with one JSON object and nothing else."""

DESCRIBE_SYSTEM = """You write the statement of a Pharo programming task. You are given the class, the methods
the solver receives, the reference implementation of the methods to implement, and the tests.
Write the statement a developer needs in order to implement the target methods correctly
WITHOUT seeing the reference or the tests: for each target method say what it receives, what it
returns (exact kind of object, order of elements, behaviour on empty or invalid input) and
every rule the tests rely on. Do not include code of the solution, do not mention the tests.
Answer with one JSON object: {"description": "..."}."""

SOLVE_SYSTEM = """You are an expert Pharo (Smalltalk) developer. Implement the requested methods for a
Pharo {pharo} image. Use only classes and messages of a standard image. "source" is the full
source of one method: the message pattern on the first line, then the body; no class name, no
">>". Keep the given message patterns. You may add helper methods to the same class.
Answer with one JSON object: {{"methods": [{{"side": "instance", "source": "..."}}]}}."""


def pattern_of(source: str) -> str:
    """The message pattern (signature) of a method source: 'at: key put: value'."""
    try:
        ast = parse_method(source)
    except ParseError:
        return (source.strip().splitlines() or [""])[0].strip()
    args = ast.get("arguments") or []
    if not args:
        return ast["selector"]
    if ":" not in ast["selector"]:
        return f"{ast['selector']} {args[0]}"
    return " ".join(f"{k}: {a}" for k, a in zip(ast["selector"].split(":"), args))


def selector_of(source: str) -> str | None:
    try:
        return parse_method(source)["selector"]
    except ParseError:
        return None


def class_definition(task: dict) -> str:
    c = task["class"]
    slots = " ".join(f"#{v}" for v in c.get("instance_variables") or [])
    return f"{c['superclass']} << #{c['name']}\n\tslots: {{ {slots} }};\n\tpackage: '{task['package']}'"


def render_prompt(task: dict) -> str:
    """The benchmark prompt: everything a model under evaluation is given."""
    name = task["class"]["name"]
    parts = [task["description"].strip(), "", "Class:", class_definition(task)]
    if task["class"].get("comment"):
        parts += ["", f'Class comment: {task["class"]["comment"]}']
    if task.get("context_methods"):
        parts += ["", "Existing methods:"]
        for m in task["context_methods"]:
            owner = name + (" class" if m["side"] == "class" else "")
            parts += [f"{owner} >> {m['source'].strip()}", ""]
    parts += ["Implement:"]
    for m in task["target_methods"]:
        parts.append(f"{name}{' class' if m['side'] == 'class' else ''} >> {pattern_of(m['source'])}")
    return "\n".join(parts)


def _methods_block(owner: str, methods: list[dict]) -> str:
    return "\n\n".join(f"{owner}{' class' if m.get('side') == 'class' else ''} >> {m['source'].strip()}" for m in methods)


def task_from_answer(answer: dict, base: dict) -> dict:
    """Canonical task record from an author answer (base carries id, operation, seed, package)."""
    return {
        **base, "title": answer["title"].strip(),
        "class": {"name": answer["class_name"].strip(), "superclass": answer["superclass"].strip() or "Object",
                  "instance_variables": [v.strip() for v in answer["instance_variables"] if v.strip()],
                  "comment": answer["class_comment"].strip()},
        "test_class": {"name": answer["test_class_name"].strip(), "superclass": "TestCase",
                       "instance_variables": [v.strip() for v in answer["test_instance_variables"] if v.strip()]},
        "context_methods": [{"side": m["side"], "source": m["source"].strip()} for m in answer["context_methods"]],
        "target_methods": [{"side": m["side"], "source": m["source"].strip()} for m in answer["target_methods"]],
        "tests": [{"source": t["source"].strip()} for t in answer["tests"]],
    }


def answer_from_task(task: dict) -> dict:
    return {"title": task["title"], "class_name": task["class"]["name"], "superclass": task["class"]["superclass"],
            "instance_variables": task["class"]["instance_variables"], "class_comment": task["class"]["comment"],
            "context_methods": task["context_methods"], "target_methods": task["target_methods"],
            "test_class_name": task["test_class"]["name"],
            "test_instance_variables": task["test_class"]["instance_variables"], "tests": task["tests"]}


class Synthesizer:
    def __init__(self, author, solvers: list, bcfg: dict, cache_dir: Path, operation_text):
        self.author, self.solvers, self.bcfg = author, solvers, bcfg
        self.operation_text = operation_text  # operation id -> definition shown to the author
        self.cache_path = Path(cache_dir) / "responses.jsonl"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self._cache[r["key"]] = r
                except ValueError:
                    continue
        self.usage = {"calls": 0, "cached": 0, "failed": 0}
        fmt = {"pharo": str(bcfg["pharo_version"])[:2], "prefix": bcfg["class_prefix"], "min_tests": bcfg["min_tests"]}
        self.author_system = AUTHOR_SYSTEM.format(**fmt)
        self.solve_system = SOLVE_SYSTEM.format(**fmt)

    # -- one cached, schema-constrained call ----------------------------------------------------
    def _ask(self, provider, role: str, system: str, user: str, schema: dict) -> dict | None:
        key = hashlib.sha256(json.dumps([PROMPT_VERSION, provider.name, provider.model, system, user]).encode()).hexdigest()
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            self.usage["cached"] += 1
            return hit["parsed"]
        c = provider.complete(system, user, schema, role)
        rec = {"key": key, "role": role, "model": provider.name, "served_model": c.served_model, "parsed": c.parsed,
               "error": c.error, "usage": c.usage, "latency_seconds": c.latency_seconds}
        with self._lock:
            self.usage["calls"] += 1
            self.usage["failed"] += c.parsed is None
            self._cache[key] = rec
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return c.parsed

    # -- author ---------------------------------------------------------------------------------
    def _request(self, seed: dict) -> str:
        return (f"Programming operation the task must exercise: {seed['operation']}\n"
                f"{self.operation_text(seed['operation'])}\n\n"
                "Seed: a real method from an open-source Pharo project that performs this operation. Use it to see\n"
                "how the operation is written in practice, then write a NEW, self-contained task about a\n"
                "different small problem. Do not copy the seed.\n\n"
                f"{seed['class_name']}{' class' if seed['side'] == 'class' else ''} >> {seed['source'].strip()}\n")

    def write_task(self, seed: dict, base: dict) -> dict | None:
        a = self._ask(self.author, "author", self.author_system, self._request(seed), TASK_SCHEMA)
        return task_from_answer(a, base) if a else None

    def repair_task(self, seed: dict, task: dict, failure: dict, attempt: int) -> dict | None:
        user = (self._request(seed) + "\nYour previous answer:\n" + json.dumps(answer_from_task(task), ensure_ascii=False)
                + f"\n\nIt was rejected by the check \"{failure['gate']}\" (attempt {attempt}):\n{failure['detail']}\n\n"
                "Correct it and answer with the complete JSON object again. If a test expected a wrong value, fix the\n"
                "test; if the solution is wrong, fix the solution. Do not weaken or remove tests.")
        a = self._ask(self.author, "repair", self.author_system, user, TASK_SCHEMA)
        return task_from_answer(a, {k: task[k] for k in ("task_id", "operation", "seed", "package")}) if a else None

    def amplify_tests(self, task: dict, survivors: list[dict], attempt: int) -> dict | None:
        """Mutation-guided test generation: ask for tests that tell each surviving mutant apart."""
        shown = "\n\n".join(f"Mutant {i} ({s['operator']}):\n{s['source'].replace(chr(13), chr(10))}"
                            for i, s in enumerate(survivors[:6], 1))
        user = ("The class and its reference solution:\n" + _methods_block(task["class"]["name"],
                                                                         task["context_methods"] + task["target_methods"])
                + "\n\nThe current tests:\n" + _methods_block(task["test_class"]["name"], task["tests"])
                + f"\n\nThese modified versions of the solution are wrong but still pass every test (attempt {attempt}):\n\n"
                + shown + "\n\nWrite additional test methods (new names, same test class, same instance variables: "
                + ", ".join(task["test_class"]["instance_variables"]) + ") that pass on the reference solution and fail "
                "on these mutants. Each \"source\" is one complete test method. If a mutant behaves exactly like the "
                "reference for every input, skip it. Answer: {\"tests\": [{\"source\": \"...\"}]}.")
        a = self._ask(self.author, "amplify", self.author_system, user, TESTS_SCHEMA)
        if not a or not a.get("tests"):
            return None
        tests = {pattern_of(t["source"]): t for t in task["tests"]}
        for t in a["tests"]:
            if t["source"].strip():
                tests[pattern_of(t["source"])] = {"source": t["source"].strip()}
        return {**task, "tests": list(tests.values())}

    # -- statement and independent solution -----------------------------------------------------
    def describe(self, task: dict, feedback: str = "") -> str | None:
        user = ("Class:\n" + class_definition(task) + f"\nComment: {task['class']['comment']}\n\nMethods the solver receives:\n"
                + (_methods_block(task["class"]["name"], task["context_methods"]) or "(none)")
                + "\n\nReference implementation of the methods to implement:\n"
                + _methods_block(task["class"]["name"], task["target_methods"])
                + "\n\nTests:\n" + _methods_block(task["test_class"]["name"], task["tests"]) + feedback)
        a = self._ask(self.author, "describe", DESCRIBE_SYSTEM, user, DESCRIPTION_SCHEMA)
        return (a or {}).get("description") or None

    def solve(self, provider, task: dict, attempt: int) -> list[dict] | None:
        user = render_prompt(task) + (f"\n\n(attempt {attempt})" if attempt > 1 else "")
        a = self._ask(provider, "solve", self.solve_system, user, SOLUTION_SCHEMA)
        if not a or not a.get("methods"):
            return None
        return [{"side": m["side"], "source": m["source"].strip()} for m in a["methods"] if m["source"].strip()]
