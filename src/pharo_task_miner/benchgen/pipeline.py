"""Benchmark generation pipeline: seeds from the mined corpus -> synthesis -> validation in the
Pharo image -> an installable, verified benchmark. The method is documented in
docs/4-benchmark.md; this module only orchestrates.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

from ..analysis.operation_rules import RuleClassifier
from ..analysis.structural_features import compute_features
from ..extraction.method_extractor import extract_repository
from ..extraction.pharo_parser_bridge import PythonBackend
from ..sampling.corpus_filter import study_repo_ids
from ..storage.provenance import now_iso, read_jsonl, write_json, write_jsonl
from . import gates as G
from .image import ensure_image, harness_digest, run_job
from .synthesis import PROMPT_VERSION, Synthesizer, render_prompt, selector_of

GENERATE_RECOMMENDATIONS = ("create_native_pharo_task", "adapt_explicitly", "transfer_to_pharo")


# ----------------------------------------------------------------------------------- seeds
def operations_to_generate(cfg, tax: dict) -> tuple[list[str], str]:
    """Which operations get tasks, and on what basis (recorded in the manifest)."""
    bcfg = cfg["benchmark_generation"]
    level3 = [c["id"] for c in tax["categories"] if c.get("level") == 3]
    wanted = bcfg.get("operations") or "auto"
    if wanted != "auto":
        names = wanted.replace(",", " ").split() if isinstance(wanted, str) else list(wanted)
        unknown = sorted(set(names) - set(level3))
        if unknown:
            raise ValueError(f"benchmark_generation.operations: not level-3 taxonomy ids: {unknown}")
        return names, "listed in benchmark_generation.operations"
    rec = cfg.output_dir / "tables" / "benchmark_recommendations.csv"
    if rec.exists():
        with rec.open(encoding="utf-8", newline="") as f:
            ops = [r["operation"] for r in csv.DictReader(f) if r.get("recommendation") in GENERATE_RECOMMENDATIONS]
        ops = [o for o in ops if o in level3]
        if ops:
            return ops, f"recommended by the benchmark coverage analysis ({rec.name}: {', '.join(GENERATE_RECOMMENDATIONS)})"
    return level3, "all level-3 operations of the taxonomy (no benchmark coverage analysis found)"


def _rank(seed: int, key) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def select_seeds(db, cfg, operations: list[str], log=print) -> list[dict]:
    """Real methods of the study corpus (``filter-corpus``; every analysed repository when it was
    never run) that carry strong rule evidence of an operation: at most one per repository, never
    a test or a duplicate, of a size a task can be modelled on. The choice is a fixed pseudo-random
    order (hash of seed and method id), so a rerun selects the same methods."""
    bcfg = cfg["benchmark_generation"]
    rseed, per_op = int(bcfg["random_seed"]), int(bcfg["tasks_per_operation"])
    lo, hi = int(bcfg["seed_min_lines"]), int(bcfg["seed_max_lines"])
    study = study_repo_ids(db)
    in_corpus = "" if study is None else (" AND method_id IN (SELECT method_id FROM methods WHERE repo_id IN (%s))"
                                          % (",".join(map(str, sorted(study))) or "NULL"))
    seeds = []
    for op in operations:
        ids = [r[0] for r in db.query("SELECT DISTINCT method_id FROM operation_evidence WHERE operation_id=?"
                                      " AND strength='strong' AND classification='rule_based'" + in_corpus, (op,))]
        ids.sort(key=lambda i: _rank(rseed, f"{op}:{i}"))
        chosen, repos = [], set()
        for start in range(0, min(len(ids), 4000), 400):
            chunk = ids[start:start + 400]
            rows = {r["method_id"]: r for r in db.query(
                "SELECT m.method_id, m.repo_id, m.class_name, m.side, m.selector, m.source, m.source_path, m.line,"
                " r.full_name, r.license, s.commit_sha FROM methods m JOIN repositories r ON r.repo_id=m.repo_id"
                " LEFT JOIN repository_snapshots s ON s.snapshot_id=m.snapshot_id"
                f" WHERE m.method_id IN ({','.join('?' * len(chunk))}) AND m.is_test=0 AND m.parse_status='ok'"
                " AND m.duplicate_of_method_id IS NULL AND COALESCE(m.in_trait,0)=0", chunk)}
            for i in chunk:
                r = rows.get(i)
                if r is None or r["repo_id"] in repos or not lo <= r["source"].count("\n") + 1 <= hi:
                    continue
                repos.add(r["repo_id"])
                chosen.append(r)
                if len(chosen) == per_op:
                    break
            if len(chosen) == per_op:
                break
        if len(chosen) < per_op:
            log(f"  {op}: only {len(chosen)} of {per_op} seed methods available")
        for k, r in enumerate(chosen, 1):
            seeds.append({"operation": op, "index": k, "method_id": r["method_id"], "repository": r["full_name"],
                          "license": r["license"], "commit_sha": r["commit_sha"], "path": r["source_path"],
                          "line": r["line"], "class_name": r["class_name"], "side": r["side"],
                          "selector": r["selector"], "source": r["source"]})
    max_tasks = int(bcfg.get("max_tasks") or 0)
    if max_tasks and len(seeds) > max_tasks:
        # keep the operations balanced: first seed of every operation, then the second, ...
        seeds.sort(key=lambda s: (s["index"], operations.index(s["operation"])))
        seeds = seeds[:max_tasks]
        seeds.sort(key=lambda s: (operations.index(s["operation"]), s["index"]))
    return seeds


def _camel(op: str) -> str:
    return "".join(p.capitalize() for p in re.split(r"[^A-Za-z0-9]+", op) if p)


def base_of(seed: dict, bcfg: dict) -> dict:
    return {"task_id": f"{seed['operation']}-{seed['index']:02d}", "operation": seed["operation"],
            "package": f"{bcfg['baseline']}-{_camel(seed['operation'])}{seed['index']:02d}",
            "seed": {k: seed[k] for k in ("method_id", "repository", "license", "commit_sha", "path", "line", "class_name",
                                          "side", "selector")} | {"source_sha256": hashlib.sha256(seed["source"].encode()).hexdigest()}}


# ----------------------------------------------------------------------------- one harness job
def job_for(task: dict, bcfg: dict, candidates: list[dict] | None = None, mutation: bool = True,
            export_dir: Path | None = None) -> dict:
    cls, tcls = task["class"], task["test_class"]
    methods = ([{"class": cls["name"], "side": m["side"], "source": m["source"], "role": "context", "protocol": "accessing"}
                for m in task.get("context_methods") or []]
               + [{"class": cls["name"], "side": m["side"], "source": m["source"], "role": "target", "protocol": "task"}
                  for m in task["target_methods"]]
               + [{"class": tcls["name"], "side": "instance", "source": t["source"], "role": "test",
                   "protocol": "tests" if t["source"].lstrip().startswith("test") else "running"} for t in task["tests"]])
    return {
        "package": task["package"], "test_class": tcls["name"], "mutation": mutation, "repeat": int(bcfg["repeat"]),
        "test_time_limit_seconds": int(bcfg["test_time_limit_seconds"]),
        "export_dir": str(export_dir) if export_dir else None,
        "classes": [{"name": cls["name"], "superclass": cls["superclass"], "comment": cls.get("comment") or "",
                     "instance_variables": cls.get("instance_variables") or [],
                     "class_variables": cls.get("class_variables") or []},
                    {"name": tcls["name"], "superclass": tcls.get("superclass") or "TestCase",
                     "instance_variables": tcls.get("instance_variables") or [], "comment": f"Tests of {cls['name']}."}],
        "methods": methods,
        "candidates": [{"name": c["name"], "methods": [{"class": cls["name"], "side": m["side"], "source": m["source"],
                                                       "protocol": "task"} for m in c["methods"]]}
                       for c in candidates or []],
    }


def operations_found(export_dir: Path, task: dict, clf: RuleClassifier) -> dict:
    """Run the mining pipeline's own parser and rule classifier on the exported Tonel package:
    {target selector: {operation id: strength}}."""
    ex = extract_repository(Path(export_dir), "benchmark", None, PythonBackend(), export_ast=False)
    cls_info = {c["name"]: c for c in ex.classes if c["kind"] != "extension"}
    wanted = {selector_of(m["source"]) for m in task["target_methods"]}
    found = {}
    for m in ex.methods:
        if m["class_name"] != task["class"]["name"] or m["parse_status"] != "ok" or m["selector"] not in wanted:
            continue
        res = clf.classify({**m, "is_test": False}, compute_features(m, cls_info.get(m["class_name"])))
        found[m["selector"]] = {op: lab["strength"] for op, lab in res.labels.items()}
    return found


# --------------------------------------------------------------------------------- one task
class Generator:
    def __init__(self, cfg, pharo: dict, synth: Synthesizer | None, tax: dict, home: Path, log=print):
        self.cfg, self.pharo, self.synth, self.tax, self.log = cfg, pharo, synth, tax, log
        self.bcfg = cfg["benchmark_generation"]
        ocfg = cfg["classification"]["operations"]
        self.clf = RuleClassifier(tax, int(ocfg.get("multiple_operations_threshold", 4)),
                                  int(ocfg.get("insufficient_context_max_nodes", 3)))
        self.work = Path(home) / "work"

    def check(self, task: dict, tag: str, candidates: list[dict] | None = None) -> tuple[list[dict], dict]:
        """Run the harness once and evaluate the execution gates (+ the operation gate)."""
        wd = self.work / task["task_id"] / tag
        ok, detail = G.well_formed(task, self.bcfg)
        if not ok:
            return [{"gate": "well_formed", "passed": False, "detail": detail}], {}
        export = wd / "export"
        result = run_job(self.cfg, self.pharo, job_for(task, self.bcfg, candidates, True, export), wd)
        gates = G.evaluate(task, result, self.bcfg)
        if G.first_failure(gates) is None and self.bcfg.get("require_operation", True):
            if not result.get("exported"):
                gates.append({"gate": "operation_preserved", "passed": False,
                              "detail": f"Tonel export failed: {result.get('export_error')}"})
            else:
                found = operations_found(export, task, self.clf)
                hit = {sel: ops[task["operation"]] for sel, ops in found.items() if task["operation"] in ops}
                gates.append({"gate": "operation_preserved", "passed": bool(hit),
                              "detail": (f"{task['operation']} found in " + ", ".join(f"{s} ({v} evidence)" for s, v in hit.items()))
                              if hit else f"the rule classifier does not find {task['operation']} in the target methods; "
                                          f"it finds: {sorted({o for ops in found.values() for o in ops}) or 'nothing'}"})
        return gates, result

    def _solve(self, task: dict, history: list) -> tuple[dict, dict | None]:
        """Independent solution from the statement alone; returns (gate, accepted candidate)."""
        cands = []
        for p in self.synth.solvers:
            for attempt in range(1, int(self.bcfg["solver_attempts"]) + 1):
                methods = self.synth.solve(p, task, attempt)
                if methods:
                    cands.append({"name": f"{p.name}#{attempt}", "model": p.name, "methods": methods})
        if not cands:
            return {"gate": "solvable", "passed": False, "detail": "no solver returned a solution"}, None
        wd = self.work / task["task_id"] / f"solve{len(history)}"
        res = run_job(self.cfg, self.pharo, job_for(task, self.bcfg, cands, False, None), wd)
        n = len(res.get("tests") or [])
        outcomes = []
        for c, r in zip(cands, res.get("candidates") or []):
            if G.candidate_passes(r, n):
                return {"gate": "solvable", "passed": True, "detail": f"solved by {c['name']} from the statement alone"}, c
            bad = r.get("errors") or [f"{t}: {v['outcome']} {v.get('detail', '')}".strip()
                                      for t, v in (r.get("tests") or {}).items() if v["outcome"] != "pass"]
            outcomes.append(f"{c['name']}: " + "; ".join(bad[:3]))
        return {"gate": "solvable", "passed": False,
                "detail": " | ".join(outcomes)[:1500] or f"harness: {res.get('stage')}"}, None

    def generate(self, seed: dict) -> dict:
        b = self.bcfg
        base = base_of(seed, b)
        history, task, gates, result = [], None, [], {}
        rounds = 1 + int(b["max_repair_rounds"]) + int(b["max_amplification_rounds"])
        repairs = amplifications = 0
        task = self.synth.write_task(seed, base)
        for i in range(rounds):
            if task is None:
                gates = [{"gate": "well_formed", "passed": False, "detail": "the author model returned no valid answer"}]
                break
            gates, result = self.check(task, f"attempt{i}")
            ff = G.first_failure(gates)
            history.append({"attempt": i, "failed": ff["gate"] if ff else None, "detail": ff["detail"] if ff else None})
            if ff is None:
                break
            survivors = G.mutation_summary(result)["survivors"] if ff["gate"] == "mutation" else []
            if survivors and amplifications < int(b["max_amplification_rounds"]):
                amplifications += 1
                nxt = self.synth.amplify_tests(task, survivors, amplifications)
            elif repairs < int(b["max_repair_rounds"]):
                repairs += 1
                nxt = self.synth.repair_task(seed, task, ff, repairs)
            else:
                break
            if nxt is None:
                break
            task = nxt
        record = {**(task or base), "gates": gates, "history": history, "repairs": repairs, "amplifications": amplifications}
        if task is None or G.first_failure(gates):
            return self._finish(record, result)
        # statement written last, from validated code; then solved by a model that sees only the statement
        feedback = ""
        for attempt in range(2):
            description = self.synth.describe(task, feedback)
            if not description:
                gates.append({"gate": "solvable", "passed": False, "detail": "no task statement was produced"})
                break
            task = {**task, "description": description}
            task["prompt"] = render_prompt(task)
            leaked = [ln.strip() for m in task["target_methods"] for ln in m["source"].splitlines()[1:]
                      if len(ln.strip()) > 30 and ln.strip() in description]
            if leaked:
                gates.append({"gate": "solvable", "passed": False,
                              "detail": f"the statement contains code of the reference solution: {leaked[0]}"})
                break
            if not b.get("require_independent_solution", True) or not self.synth.solvers:
                break
            gate, cand = self._solve(task, history)
            history.append({"attempt": f"solve{attempt}", "failed": None if gate["passed"] else "solvable", "detail": gate["detail"]})
            if gate["passed"] or attempt == 1:
                gates.append(gate)
                if cand:
                    task["independent_solution"] = {"model": cand["model"], "methods": cand["methods"]}
                break
            feedback = ("\n\nA developer who read your previous statement wrote a solution that failed:\n" + gate["detail"]
                        + "\nRewrite the statement so that it states precisely what was missing or ambiguous.")
        record = {**task, "gates": gates, "history": history, "repairs": repairs, "amplifications": amplifications}
        return self._finish(record, result)

    def verify(self, task: dict) -> dict:
        """Re-validate a stored task without any model (benchmark-validate)."""
        task = {k: v for k, v in task.items() if k not in ("gates", "status", "rejected_by", "reason", "validation")}
        cands = ([{"name": "independent_solution", "methods": task["independent_solution"]["methods"]}]
                 if task.get("independent_solution") else [])
        gates, result = self.check(task, "verify", cands)
        if G.first_failure(gates) is None and cands:
            r = (result.get("candidates") or [{}])[0]
            ok = G.candidate_passes(r, len(result.get("tests") or []))
            gates.append({"gate": "solvable", "passed": ok, "detail": "stored independent solution re-run: "
                          + ("passes all tests" if ok else "does not pass")})
        return self._finish({**task, "gates": gates}, result)

    def _finish(self, record: dict, result: dict) -> dict:
        ff = G.first_failure(record["gates"])
        ms = G.mutation_summary(result) if result else {}
        record["validation"] = {
            "image": result.get("image"), "tests": result.get("tests"),
            "mutation": {k: ms.get(k) for k in ("evaluated", "detected", "survived", "score")} | {
                "survivors": [{"operator": s["operator"], "selector": s["selector"]} for s in ms.get("survivors") or []]},
            "fail_without_solution": sum(1 for v in (result.get("all_stubbed_run") or {}).values() if v["outcome"] != "pass"),
        }
        record["status"] = "rejected" if ff else "accepted"
        record["rejected_by"], record["reason"] = (ff["gate"], ff["detail"]) if ff else (None, None)
        return record


# ------------------------------------------------------------------------- the whole benchmark
def _unique(records: list[dict], bcfg: dict, seeds: dict):
    names, hashes = {}, {}
    for r in sorted((r for r in records if r["status"] == "accepted"), key=lambda r: r["task_id"]):
        why = None
        for n in (r["class"]["name"], r["test_class"]["name"]):
            if n in names:
                why = f"class name {n} already used by {names[n]}"
        h = G.solution_hash(r)
        if not why and h in hashes:
            why = f"same solution as {hashes[h]}"
        src = seeds.get(r["task_id"])
        if not why and src:
            sim = max(G.similarity(m["source"], src) for m in r["target_methods"])
            if sim > float(bcfg["max_seed_similarity"]):
                why = f"solution too close to its seed method (similarity {sim:.2f})"
        r["gates"] = [g for g in r["gates"] if g["gate"] not in ("unique", "installs_together")]
        r["gates"].append({"gate": "unique", "passed": why is None, "detail": why or "distinct names and solution"})
        if why:
            r["status"], r["rejected_by"], r["reason"] = "rejected", "unique", why
        else:
            names[r["class"]["name"]] = names[r["test_class"]["name"]] = hashes[h] = r["task_id"]


def _tonel_escape(s: str) -> str:
    return s.replace("'", "''")


def write_repository(cfg, accepted: list[dict], home: Path) -> Path:
    """The installable benchmark: one Tonel package per task (as exported by the image) and a
    Metacello baseline that loads them all."""
    bcfg = cfg["benchmark_generation"]
    repo = home / bcfg["baseline"]
    if repo.exists():
        shutil.rmtree(repo)
    src = repo / "src"
    src.mkdir(parents=True)
    for r in accepted:
        # the package as the image wrote it in the task's last (accepted) validation run
        exported = sorted((home / "work" / r["task_id"]).glob(f"*/export/{r['package']}"), key=lambda p: p.stat().st_mtime)
        shutil.copytree(exported[-1], src / r["package"])
    (src / ".properties").write_text("{\n\t#format : #tonel\n}\n", encoding="utf-8")
    (repo / ".project").write_text("{\n\t'srcDirectory' : 'src'\n}\n", encoding="utf-8")
    base = f"BaselineOf{bcfg['baseline']}"
    (src / base).mkdir()
    (src / base / "package.st").write_text(f"Package {{ #name : '{base}' }}\n", encoding="utf-8")
    packages = "\n".join(f"\t\t\tspec package: '{_tonel_escape(r['package'])}'." for r in accepted)
    (src / base / f"{base}.class.st").write_text(
        f'"\nI load the {len(accepted)} tasks of the {bcfg["baseline"]} benchmark.\n"\n'
        f"Class {{\n\t#name : '{base}',\n\t#superclass : 'BaselineOf',\n\t#category : '{base}',\n\t#package : '{base}'\n}}\n\n"
        f"{{ #category : 'baselines' }}\n{base} >> baseline: spec [\n\t<baseline>\n"
        f"\tspec for: #common do: [\n{packages}\n\t\t]\n]\n", encoding="utf-8")
    (repo / "README.md").write_text(
        f"# {bcfg['baseline']}\n\n{len(accepted)} Pharo programming tasks generated and validated automatically by "
        "PharoTaskMiner (see `docs/4-benchmark.md` in that project).\n\n"
        "```smalltalk\nMetacello new\n\tbaseline: '" + bcfg["baseline"] + "';\n\trepository: 'tonel://<this directory>/src';\n\tload.\n```\n\n"
        "Each package holds one class with its reference solution and one test class. The task statements, the "
        "provenance and the validation evidence of every task are in `tasks.jsonl`.\n", encoding="utf-8")
    return repo


def install_together(cfg, pharo: dict, accepted: list[dict], repo: Path, home: Path) -> None:
    """Load the whole benchmark through its baseline in one clean image and run every test class."""
    bcfg = cfg["benchmark_generation"]
    job = {"repository": str(repo / "src"), "baseline": bcfg["baseline"],
           "test_classes": [r["test_class"]["name"] for r in accepted]}
    res = run_job(cfg, pharo, job, home / "work" / "_install", "install.st",
                  timeout=int(bcfg["job_timeout_seconds"]) * 3)
    by = {c["test_class"]: c for c in res.get("classes") or []}
    for r in accepted:
        c = by.get(r["test_class"]["name"])
        n = len((r.get("validation") or {}).get("tests") or [])
        if res.get("stage") != "done":
            ok, detail = False, f"the benchmark did not load: {res.get('detail') or res.get('stage')}"[:400]
        elif not c or c.get("missing"):
            ok, detail = False, "test class missing after loading the baseline"
        else:
            ok = not c["failed"] and c["passed"] == c["run"] == n
            detail = f"{c['passed']}/{c['run']} tests pass after loading all {len(accepted)} tasks" + (
                f"; failing: {', '.join(c['failed'])}" if c["failed"] else "") + ("" if c["run"] == n else f" (expected {n} tests)")
        r["gates"].append({"gate": "installs_together", "passed": ok, "detail": detail})
        if not ok:
            r["status"], r["rejected_by"], r["reason"] = "rejected", "installs_together", detail


def finalize(cfg, pharo: dict, records: list[dict], seeds: dict, manifest: dict, home: Path, report: Path, log=print) -> dict:
    bcfg = cfg["benchmark_generation"]
    _unique(records, bcfg, seeds)
    repo = None
    for _ in range(3):  # a task dropped by the joint installation changes the set: check again
        accepted = sorted((r for r in records if r["status"] == "accepted"), key=lambda r: r["task_id"])
        if not accepted:
            break
        for r in accepted:
            r["gates"] = [g for g in r["gates"] if g["gate"] != "installs_together"]
        repo = write_repository(cfg, accepted, home)
        install_together(cfg, pharo, accepted, repo, home)
        if all(r["status"] == "accepted" for r in accepted):
            break
    accepted = sorted((r for r in records if r["status"] == "accepted"), key=lambda r: r["task_id"])
    rejected = sorted((r for r in records if r["status"] != "accepted"), key=lambda r: r["task_id"])
    if not accepted and (home / bcfg["baseline"]).exists():
        shutil.rmtree(home / bcfg["baseline"])
    write_jsonl(home / "tasks.jsonl", accepted)
    write_jsonl(home / "rejected.jsonl", rejected)
    if repo and accepted:
        shutil.copy2(home / "tasks.jsonl", repo / "tasks.jsonl")
    # the same tasks in the format `pharo-miner benchmark-coverage --input` reads
    write_json(home / "benchmark_tasks.json", {"tasks": [
        {"benchmark": bcfg["baseline"], "task_id": r["task_id"], "description": r.get("description") or r.get("title"),
         "operations": [r["operation"]], "structure": ["single_class"], "required_capabilities": [],
         "transfer_status": "native_pharo", "difficulty": "", "preserves_capability": True,
         "origin": f"synthesised from {r['seed']['repository']}@{(r['seed'].get('commit_sha') or '')[:12]}"
         if r.get("seed") else "synthesised"} for r in accepted]})
    funnel, remaining = [], len(records)
    by_gate = Counter(r["rejected_by"] for r in rejected)
    for gid, text in G.GATES:
        remaining -= by_gate.get(gid, 0)
        funnel.append({"gate": gid, "checks": text, "rejected": by_gate.get(gid, 0), "remaining": remaining})
    scores = [r["validation"]["mutation"]["score"] for r in accepted if r["validation"]["mutation"].get("score") is not None]
    per_op = Counter(r["operation"] for r in accepted)
    manifest = {
        **manifest, "finished_at": now_iso(), "image": pharo["info"], "harness_sha256": harness_digest(cfg),
        "prompt_version": PROMPT_VERSION,
        "thresholds": {k: bcfg[k] for k in ("min_tests", "repeat", "min_mutation_score", "min_mutants", "max_target_methods",
                                            "max_seed_similarity", "require_operation", "require_independent_solution",
                                            "max_repair_rounds", "max_amplification_rounds", "solver_attempts",
                                            "test_time_limit_seconds")},
        "candidates": len(records), "accepted": len(accepted), "rejected": len(rejected), "funnel": funnel,
        "accepted_per_operation": dict(sorted(per_op.items())),
        "mutation_score": ({"mean": round(sum(scores) / len(scores), 4), "min": min(scores),
                            "all_mutants_detected": sum(s == 1.0 for s in scores)} if scores else None),
        "tests_per_task": (round(sum(len(r["validation"]["tests"]) for r in accepted) / len(accepted), 2) if accepted else None),
        "repaired_tasks": sum(1 for r in accepted if r.get("repairs")),
        "tasks_with_added_tests": sum(1 for r in accepted if r.get("amplifications")),
    }
    write_json(home / "manifest.json", manifest)
    write_report(cfg, manifest, accepted, rejected, report)
    log(f"Benchmark: {len(accepted)} tasks accepted, {len(rejected)} rejected of {len(records)} candidates")
    for f in funnel:
        if f["rejected"]:
            log(f"  rejected by {f['gate']}: {f['rejected']}")
    return manifest


def write_report(cfg, manifest: dict, accepted: list[dict], rejected: list[dict], out: Path):
    bcfg = cfg["benchmark_generation"]
    L = [f"# Benchmark generation: {bcfg['baseline']}", "",
         f"Generated {manifest['finished_at']}. Method: `docs/4-benchmark.md`.", "",
         f"- Image: {manifest['image'].get('system_version')} (sha256 `{manifest['image'].get('image_sha256', '')[:16]}`), "
         f"MuTalk {manifest['image'].get('mutalk_version')}",
         f"- Author model: {manifest.get('author') or 'none (validation only)'}; independent solvers: "
         f"{', '.join(manifest.get('solvers') or []) or 'none'}",
         f"- Operations: {manifest.get('operations_basis', '')}",
         f"- **{manifest['accepted']} tasks accepted** of {manifest['candidates']} candidates", ""]
    if not manifest.get("solvers") and manifest.get("author"):
        L += ["> No independent solver was configured: the gate \"solvable\" was not applied. The statements are "
              "therefore not verified against the tests.", ""]
    L += ["## Filtering funnel", "", "| Gate | What it checks | Rejected | Remaining |", "|---|---|---:|---:|"]
    L += [f"| {f['gate']} | {f['checks']} | {f['rejected']} | {f['remaining']} |" for f in manifest["funnel"]]
    ms = manifest.get("mutation_score")
    if ms:
        L += ["", "## Test strength of the accepted tasks", "",
              f"- Mean mutation score: {ms['mean']:.2f} (minimum {ms['min']:.2f}; required {bcfg['min_mutation_score']})",
              f"- Tasks whose tests detect every mutant: {ms['all_mutants_detected']} of {manifest['accepted']}",
              f"- Tests per task (mean): {manifest['tests_per_task']}",
              f"- Tasks repaired after an image error: {manifest['repaired_tasks']}; "
              f"tasks that received tests against surviving mutants: {manifest['tasks_with_added_tests']}"]
    if accepted:
        L += ["", "## Accepted tasks", "", "| Task | Operation | Class | Tests | Mutants detected | Seed repository |",
              "|---|---|---|---:|---:|---|"]
        for r in accepted:
            m = r["validation"]["mutation"]
            L.append(f"| {r['task_id']} | {r['operation']} | {r['class']['name']} | {len(r['validation']['tests'])} | "
                     f"{m.get('detected')}/{m.get('evaluated')} | {(r.get('seed') or {}).get('repository', '')} |")
    if rejected:
        L += ["", "## Rejected candidates", "", "| Task | Operation | Rejected by | Reason |", "|---|---|---|---|"]
        for r in rejected:
            reason = (r.get("reason") or "").replace("|", "/").replace("\n", " ")
            L.append(f"| {r['task_id']} | {r['operation']} | {r['rejected_by']} | {reason[:200]} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------ entry points
def run_generation(cfg, db, tax: dict, author, solvers: list, log=print, limit: int = 0) -> dict:
    bcfg = cfg["benchmark_generation"]
    home = cfg.resolve(bcfg["dir"])
    pharo = ensure_image(cfg, log)
    ops, basis = operations_to_generate(cfg, tax)
    log(f"Operations: {len(ops)} ({basis})")
    seeds = select_seeds(db, cfg, ops, log)
    if limit:
        seeds = seeds[:limit]
    log(f"Seed methods: {len(seeds)} from {len({s['repository'] for s in seeds})} repositories")
    cats = {c["id"]: c for c in tax["categories"]}

    def operation_text(op):
        c = cats[op]
        parent = cats.get(c.get("parent"), {})
        return (f"{c.get('name', op)}: {c.get('definition', '')}"
                + (f" (part of: {parent.get('name')})" if parent else "")
                + (f"\nIncludes: {c['include']}" if c.get("include") else "")
                + (f"\nExcludes: {c['exclude']}" if c.get("exclude") else ""))

    synth = Synthesizer(author, solvers, bcfg, home / "llm_cache", operation_text)
    gen = Generator(cfg, pharo, synth, tax, home, log)
    state = home / "candidates.jsonl"
    signature = hashlib.sha256(json.dumps([PROMPT_VERSION, author.name, [s.name for s in solvers], harness_digest(cfg),
                                           pharo["info"]["image_sha256"],
                                           {k: v for k, v in bcfg.items() if k not in ("concurrency", "max_tasks")}],
                                          sort_keys=True, default=str).encode()).hexdigest()[:16]
    done = {r["task_id"]: r for r in read_jsonl(state) if r.get("run_signature") == signature}
    todo = [s for s in seeds if base_of(s, bcfg)["task_id"] not in done]
    if done:
        log(f"Resuming: {len(seeds) - len(todo)} candidates already decided")

    def one(seed):
        rec = gen.generate(seed)
        rec["run_signature"] = signature
        return rec

    with ThreadPoolExecutor(max_workers=max(1, int(bcfg["concurrency"]))) as pool:
        for i, rec in enumerate(pool.map(one, todo), 1):
            done[rec["task_id"]] = rec
            with state.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            log(f"  [{i}/{len(todo)}] {rec['task_id']}: {rec['status']}"
                + (f" ({rec['rejected_by']}: {(rec['reason'] or '')[:110]})" if rec["status"] != "accepted" else
                   f" (mutants {rec['validation']['mutation']['detected']}/{rec['validation']['mutation']['evaluated']})"))
    wanted = {base_of(s, bcfg)["task_id"] for s in seeds}
    records = [r for t, r in done.items() if t in wanted]
    manifest = {"started_at": now_iso(), "author": author.name, "solvers": [s.name for s in solvers],
                "operations_basis": basis, "operations": ops, "seeds": len(seeds), "run_signature": signature,
                "model_calls": synth.usage}
    return finalize(cfg, pharo, records, {base_of(s, bcfg)["task_id"]: s["source"] for s in seeds}, manifest, home,
                    cfg.output_dir / "reports" / "benchmark_generation.md", log)


def run_validation(cfg, tax: dict, tasks: list[dict], log=print) -> dict:
    """Re-verify stored tasks in the image, with no model involved. Everything is written to
    <dir>/verification, so the generated benchmark itself is never modified."""
    bcfg = cfg["benchmark_generation"]
    pharo = ensure_image(cfg, log)
    home = cfg.resolve(bcfg["dir"]) / "verification"
    if (home / "work").exists():
        shutil.rmtree(home / "work")
    gen = Generator(cfg, pharo, None, tax, home, log)
    records = []
    with ThreadPoolExecutor(max_workers=max(1, int(bcfg["concurrency"]))) as pool:
        for i, rec in enumerate(pool.map(gen.verify, tasks), 1):
            records.append(rec)
            log(f"  [{i}/{len(tasks)}] {rec['task_id']}: {rec['status']}"
                + (f" ({rec['rejected_by']}: {(rec['reason'] or '')[:110]})" if rec["status"] != "accepted" else ""))
    manifest = {"started_at": now_iso(), "author": None, "solvers": [], "operations_basis": "re-validation of stored tasks"}
    return finalize(cfg, pharo, records, {}, manifest, home, cfg.output_dir / "reports" / "benchmark_verification.md", log)
