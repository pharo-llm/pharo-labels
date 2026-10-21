"""Independent validation of benchmark tasks: behaviour contracts, independent tests, mutant validity.

A task whose reference, tests and statement all come from one model can be consistently wrong. So:

* **Behaviour contract** (phase 11, before any task): for each selected behaviour, a model of another family
  than the author writes, from the taxonomy definition only, what ANY implementation of the behaviour must do:
  input, output, 3–6 observable properties (``P1``…), edge cases, what is excluded. Every task of the
  behaviour is written against it.
* **Independent tests** (phase 12): oracle models (families ≠ author) see only what a solver sees (statement,
  class, given methods, signatures) and the contract — never the reference nor the author's tests — and
  write SUnit tests, one or more per contract property. Run on the reference in a clean image, each test is:
  ``agrees`` (passes on every repeated run), ``counterexample`` (fails: the reference or the test is wrong —
  quarantined for the audit), ``flaky``, ``vacuous`` (passes with every target stubbed: tests nothing),
  or ``invalid`` (does not compile, or calls a method the task does not have).
* **Behaviour alignment**: a contract property is ``covered`` when an agreeing, non-vacuous test checks it,
  ``violated`` when a counterexample does, ``untested`` otherwise.
* **Mutant validity**: every MuTalk mutant of the target methods is re-run in a clean image on the task's
  tests and on the independent tests: ``killed`` (reproducibly), ``survived_detectable`` (the task's tests
  miss a fault the independent tests see: a test gap), ``possibly_equivalent`` (no test sees a difference;
  equivalence is undecidable in general, so it is never assumed), ``unstable`` (the verdict differs between
  images), ``terminated`` (crash or time-out under MuTalk), ``invalid`` (does not compile).
  A bug-repair mutant must compile, be killed reproducibly by a failing assertion or error (not only a
  time-out) and violate the contract (fail an agreeing independent test).
"""
from __future__ import annotations

import random
import re

CONTRACT_VERSION = "contract-v1"
ORACLE_VERSION = "oracle-v1"

_LINES = {"type": "array", "items": {"type": "string"}, "minItems": 1}

CONTRACT_SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"}, "input": {"type": "string"}, "output": {"type": "string"},
    "properties": {"type": "array", "minItems": 3, "maxItems": 6, "items": {
        "type": "object", "properties": {"id": {"type": "string"}, "statement": {"type": "string", "minLength": 10}},
        "required": ["id", "statement"]}},
    "edge_cases": {"type": "array", "items": {"type": "string"}},
    "excluded": {"type": "string"}},
    "required": ["name", "input", "output", "properties", "edge_cases", "excluded"]}

CONTRACT_PROMPT = """You write the BEHAVIOUR CONTRACT of one programming behaviour, before any task exists for it. The contract defines the behaviour independently of any programming language, application domain or example. Every benchmark task later written for this behaviour must satisfy it, and tests will be derived from it.
- input: what the behaviour operates on (kinds of values, not a concrete domain).
- output: what it returns or changes, observable from outside.
- properties: 3 to 6 REQUIRED properties, ids P1, P2, ... Each one observable and testable by calling the code on chosen inputs and checking the result (e.g. "every returned element satisfies the predicate", "the relative order of the kept elements is preserved"). Never a property about how it is implemented (which loop, which method).
- edge_cases: inputs the behaviour must handle (empty, boundary, invalid) and what must happen then.
- excluded: what does NOT count as this behaviour (a close but different behaviour)."""

ORACLE_SCHEMA = {"type": "object", "properties": {
    "tests": {"type": "array", "minItems": 3, "maxItems": 12, "items": {
        "type": "object", "properties": {"property": {"type": "string"}, "lines": _LINES}, "required": ["property", "lines"]}},
    "untestable": {"type": "array", "items": {
        "type": "object", "properties": {"property": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["property", "reason"]}}},
    "required": ["tests", "untestable"]}

ORACLE_PROMPT = """You write INDEPENDENT tests for a Pharo 13 programming task. You see what a solver sees (the statement, the class, its given methods, the message patterns of the methods to implement) and the behaviour contract the task must satisfy. You never see the solution. Your tests must accept ANY correct implementation, so they follow only the statement and the contract: never test what the statement leaves open (an order it does not specify, exact error messages, the class of an error it does not name).
Rules:
- Tests are methods of a subclass of PBTestCase whose instance variable `task` already holds a fresh instance of the class (setUp does task := TheClass new). Use assert:, assert:equals:, deny:, should:raise:.
- Call only the class's given methods, the methods to implement, and standard Pharo 13 classes and messages. No file, network, clock, random value, Smalltalk globals.
- Pharo syntax: temporaries are declared once as `| a b |` on the line after the message pattern; statements end with a period; blocks are `[ :x | ... ]`; literal arrays `#(1 2 3)`, dynamic arrays `{ a. b }`.
- Each test is a unary method named test... that checks one thing.
- For EACH contract property, at least one test that checks it on this task, with "property" set to its id (P1, ...). Tests of the statement that no property covers have "property": "spec". A property that cannot be checked through this task's methods goes in "untestable" with the reason.
- Every method is given as "lines": one array element per line of code; the first line is the message pattern (e.g. testEmptyInput). No class name, no ">>"."""


def contract_id(behavior_id: str) -> str:
    return f"CONTRACT-{behavior_id}"


def contract_user(beh: dict, path: str) -> str:
    return (f"BEHAVIOUR {beh.get('id')} {beh.get('name')} ({path})\nDefinition: {beh.get('definition')}\n"
            + (f"Use when: {'; '.join(beh.get('include') or [])}\n" if beh.get("include") else "")
            + (f"Do not use when: {'; '.join(beh.get('exclude') or [])}\n" if beh.get("exclude") else "")
            + "\nWrite the contract as JSON.")


def normalize_contract(data: dict, behavior_id: str) -> dict:
    """Property ids made P1..Pn in order (models number them their own way)."""
    props = [{"id": f"P{i}", "statement": str(p.get("statement", "")).strip()}
             for i, p in enumerate(data.get("properties") or [], 1) if str(p.get("statement", "")).strip()]
    return {"contract_id": contract_id(behavior_id), "behavior_id": behavior_id, "name": data.get("name"),
            "input": data.get("input"), "output": data.get("output"), "properties": props,
            "edge_cases": list(data.get("edge_cases") or []), "excluded": data.get("excluded")}


def contract_text(c: dict | None) -> str:
    if not c:
        return ""
    lines = [f"{c['contract_id']} — {c.get('name')}", f"Input: {c.get('input')}", f"Output: {c.get('output')}",
             "Required properties:"] + [f"  {p['id']}. {p['statement']}" for p in c["properties"]]
    if c.get("edge_cases"):
        lines += ["Edge cases:"] + [f"  - {e}" for e in c["edge_cases"]]
    if c.get("excluded"):
        lines.append(f"Excluded (not this behaviour): {c['excluded']}")
    return "\n".join(lines)


def contract_markdown(c: dict) -> str:
    return f"# {c['contract_id']}\n\n```\n{contract_text(c)}\n```\n"


# ---------------------------------------------------------------------------------------------- independent tests
def _src(m: dict) -> str:
    return "\n".join(str(x).rstrip() for x in m.get("lines") or []).strip()


def oracle_user(pub: dict, contract: dict | None) -> str:
    """What an oracle model sees: the solver's view of the task (``public_view``) and the contract."""
    ivars = " ".join(pub["instance_variables"])
    lines = ["STATEMENT:", pub["statement"].strip(), "",
             f"Pharo {pub['target_pharo_version']}. Class {pub['class']} (subclass of {pub['superclass']}"
             + (f", instance variables: {ivars}" if ivars else "") + ")."]
    if pub["context_methods"]:
        lines += ["", "Its given methods:"] + [f"```smalltalk\n{m}\n```" for m in pub["context_methods"]]
    lines += ["", "Methods a solver implements (message patterns):"] + [f"- {m['signature']}" for m in pub["target_methods"]]
    lines += ["", "BEHAVIOUR CONTRACT:", contract_text(contract)] if contract else \
             ["", "(No behaviour contract: test the statement only, every test with \"property\": \"spec\".)"]
    return "\n".join(lines + ["", "Write the tests as JSON."])


def oracle_tests(answers: list[tuple[str, dict]], contract: dict | None) -> tuple[list[dict], list[dict]]:
    """The tests of every oracle answer, renamed ``testO<oracle><n>`` (unique, provenance in the name), and the
    answers' rejected items. ``answers``: (oracle name, answer data)."""
    props = {p["id"] for p in (contract or {}).get("properties") or []}
    tests, rejected = [], []
    for name, data in answers:
        tag = re.sub(r"[^A-Za-z0-9]", "", name).capitalize() or "X"
        for i, t in enumerate((data or {}).get("tests") or [], 1):
            source = _src(t)
            lines = source.splitlines()
            head = lines[0].split('"')[0].strip() if lines else ""
            prop = str(t.get("property", "spec")).strip().upper()
            prop = prop if prop in props else "SPEC"
            if not re.fullmatch(r"test\w*", head):
                rejected.append({"oracle": name, "source": source, "reason": "not a unary test method"})
                continue
            sel = f"testO{tag}{i}"
            tests.append({"selector": sel, "oracle": name, "property": prop if prop != "SPEC" else "spec",
                          "original_selector": head, "source": "\n".join([sel] + lines[1:])})
    return tests, rejected


def oracle_task(task: dict, tests: list[dict]) -> dict:
    """The task with the independent tests as its test class (same setUp, its own class name)."""
    cls = task["class"]["name"]
    setup = next(t for t in task["tests"] if t["source"].startswith("setUp"))
    return {**task, "test_class": {"name": cls + "OTest", "superclass": "PBTestCase"},
            "tests": [setup] + [{"source": t["source"]} for t in tests]}


def interface_misuse(detail: str, cls: str) -> bool:
    """The test sent the task a message it does not have: the test is invalid, not the reference."""
    return bool(re.search(rf"{re.escape(cls)}\b.*did not understand|doesNotUnderstand.*{re.escape(cls)}", detail or ""))


def classify_tests(tests: list[dict], first: dict, cls: str) -> dict:
    """{selector: status, detail} after the first run on the reference (``first``: harness result)."""
    dropped = {d.get("source", "").split("\n", 1)[0].strip(): d for d in first.get("dropped") or []}
    runs = first.get("reference_runs") or []
    out = {}
    for t in tests:
        sel = t["selector"]
        if sel in dropped:
            out[sel] = {"status": "invalid", "detail": f"{dropped[sel].get('kind')}: {dropped[sel].get('detail')}"}
            continue
        outcomes = [r[sel] for r in runs if sel in r]
        if not outcomes:
            out[sel] = {"status": "invalid", "detail": f"not run (stage {first.get('stage')})"}
        elif all(o.get("outcome") == "pass" for o in outcomes):
            out[sel] = {"status": "agrees", "detail": ""}
        elif any(o.get("outcome") == "pass" for o in outcomes):
            out[sel] = {"status": "flaky", "detail": "passes on some runs only"}
        else:
            bad = next(o for o in outcomes if o.get("outcome") != "pass")
            st = "invalid" if interface_misuse(bad.get("detail", ""), cls) else "counterexample"
            out[sel] = {"status": st, "detail": f"{bad.get('outcome')}: {bad.get('detail', '')}"}
    return out


def confirm_tests(status: dict, second: dict, repeat: int) -> dict:
    """The agreeing tests re-run alone (``second``, ``repeat`` times): not stable → flaky; passing with every
    target stubbed → vacuous."""
    runs = second.get("reference_runs") or []
    stub = second.get("all_stubbed_run") or {}
    for sel, s in status.items():
        if s["status"] != "agrees":
            continue
        outs = [r.get(sel, {}).get("outcome") for r in runs]
        if len(runs) < repeat or any(o != "pass" for o in outs):
            status[sel] = {"status": "flaky", "detail": "not passing on every repeated run"}
        elif stub.get(sel, {}).get("outcome") == "pass":
            status[sel] = {"status": "vacuous", "detail": "passes with every target method stubbed"}
    return status


def useful(status: dict) -> list[str]:
    return [sel for sel, s in status.items() if s["status"] == "agrees"]


def alignment(contract: dict | None, tests: list[dict], status: dict, untestable: list[dict]) -> dict:
    """Per contract property: covered / violated / untested; overall status and coverage."""
    if not contract or not contract.get("properties"):
        return {"status": "no_contract", "coverage": None, "properties": {}}
    props = {}
    for p in contract["properties"]:
        mine = [t for t in tests if t["property"] == p["id"]]
        sts = [status.get(t["selector"], {}).get("status") for t in mine]
        if "counterexample" in sts:
            props[p["id"]] = "violated"
        elif "agrees" in sts:
            props[p["id"]] = "covered"
        else:
            why = next((u.get("reason") for u in untestable if str(u.get("property", "")).upper() == p["id"]), None)
            props[p["id"]] = "untested" + (f" (declared untestable: {why})" if why else "")
    cov = sum(v == "covered" for v in props.values()) / len(props)
    st = ("violated" if any(v == "violated" for v in props.values())
          else "verified" if cov == 1 else "partial")
    return {"status": st, "coverage": round(cov, 3), "properties": props}


# ---------------------------------------------------------------------------------------------- mutants
def fix_newlines(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n")


def mutants_of(result: dict, task: dict, cap: int) -> list[dict]:
    """MuTalk's mutants of the target methods, numbered, in a stable order (at most ``cap`` re-checked)."""
    sels = {m["selector"] for m in task["target_methods"]}
    ms = [m for m in (result.get("mutation") or {}).get("mutants", []) if m.get("selector") in sels and m.get("source")]
    ms.sort(key=lambda m: (m["selector"], m["operator"], m["source"]))
    out = []
    for i, m in enumerate(ms):
        out.append({"mutant_id": f"m{i:03d}", "operator": m["operator"], "selector": m["selector"],
                    "mutalk_status": m.get("status"), "source": fix_newlines(m["source"]), "rechecked": i < cap})
    return out


def mutant_candidate(task: dict, m: dict) -> dict:
    """The mutated method with the OTHER target methods as in the reference (a candidate starts from stubs)."""
    cls = task["class"]["name"]
    methods = [{"class": cls, "side": "instance", "source": m["source"] if t["selector"] == m["selector"] else t["source"]}
               for t in task["target_methods"]]
    return {"name": m["mutant_id"], "methods": methods}


def _failing(tests: dict, meta: dict | None = None) -> list[dict]:
    out = []
    for sel, t in (tests or {}).items():
        if t.get("outcome") != "pass":
            row = {"test": sel, "outcome": t.get("outcome"), "detail": t.get("detail", "")}
            if meta and sel in meta:
                row["property"] = meta[sel]["property"]
            out.append(row)
    return out


def kill_kind(failing: list[dict]) -> str | None:
    kinds = {f["outcome"] for f in failing}
    return "failure" if "failure" in kinds else "error" if "error" in kinds else "timeout" if kinds else None


def classify_mutants(mutants: list[dict], task_cands: dict, oracle_cands: dict, oracle_sel: list[str],
                     oracle_meta: dict) -> list[dict]:
    """One record per mutant (see the module docstring for the classes). ``task_cands`` / ``oracle_cands``:
    harness candidate results by name, on the task's tests / on the agreeing independent tests."""
    out = []
    for m in mutants:
        rec = {**m, "task_tests_failing": [], "oracle_tests_failing": [], "kill_kind": None, "contract_violation": None}
        tc = task_cands.get(m["mutant_id"])
        if m["mutalk_status"] == "terminated":
            rec["classification"] = "terminated"
        elif not m["rechecked"] or tc is None:
            rec["classification"] = "not_rechecked"
        elif tc.get("errors"):
            rec["classification"] = "invalid"
            rec["errors"] = tc["errors"]
        else:
            fail = _failing(tc.get("tests"))
            rec["task_tests_failing"] = fail
            rec["kill_kind"] = kill_kind(fail)
            oc = oracle_cands.get(m["mutant_id"])
            if oc is not None and not oc.get("errors") and oracle_sel:
                ofail = [f for f in _failing(oc.get("tests"), oracle_meta) if f["test"] in oracle_sel]
                rec["oracle_tests_failing"] = ofail
                rec["contract_violation"] = bool(ofail)
            killed_here = bool(fail)
            if killed_here != (m["mutalk_status"] == "killed"):
                rec["classification"] = "unstable"
            elif killed_here:
                rec["classification"] = "killed"
            elif rec["contract_violation"]:
                rec["classification"] = "survived_detectable"
            else:
                rec["classification"] = "possibly_equivalent"
        rec["bug_repair_eligible"] = (rec["classification"] == "killed" and rec["kill_kind"] in ("failure", "error")
                                      and rec["contract_violation"] is True)
        out.append(rec)
    return out


def mutation_summary(records: list[dict]) -> dict:
    """Counts per class and the mutation score bounds: killed / (killed + every survivor) (no survivor assumed
    equivalent) and killed / (killed + detectable survivors) (every undetected survivor taken as equivalent)."""
    n = {}
    for r in records:
        n[r["classification"]] = n.get(r["classification"], 0) + 1
    k, sd, pe = n.get("killed", 0), n.get("survived_detectable", 0), n.get("possibly_equivalent", 0)
    return {"counts": n, "valid_mutants": k + sd + pe, "killed_mutants": k,
            "mutation_score_lower": round(k / (k + sd + pe), 3) if k + sd + pe else None,
            "mutation_score_upper": round(k / (k + sd), 3) if k + sd else None,
            "bug_repair_eligible": sum(r["bug_repair_eligible"] for r in records)}


def pick_bug(records: list[dict]) -> dict | None:
    """The bug-repair mutant: killed by a failing assertion rather than an error, and by the FEWEST tests (a
    precise fault, like an off-by-one, rather than an emptied method every test sees)."""
    ok = [r for r in records if r["bug_repair_eligible"]]
    ok.sort(key=lambda r: (r["kill_kind"] != "failure", len(r["task_tests_failing"]), r["operator"], len(r["source"]), r["source"]))
    return ok[0] if ok else None


# ---------------------------------------------------------------------------------------------- audit
def audit_sample(tasks: list[dict], n: int, seed: int, key: str = "category_id") -> list[str]:
    """A random sample of ``n`` task ids stratified by ``key``: shuffled within each stratum, then taken in turn
    from each stratum, so every category is represented before any gets a second task."""
    rng = random.Random(seed)
    groups = {}
    for t in sorted(tasks, key=lambda t: t["task_id"]):
        groups.setdefault(str(t.get(key)), []).append(t["task_id"])
    for g in groups.values():
        rng.shuffle(g)
    order, i = [], 0
    while len(order) < min(n, len(tasks)):
        for g in sorted(groups):
            if i < len(groups[g]) and len(order) < n:
                order.append(groups[g][i])
        i += 1
    return order
