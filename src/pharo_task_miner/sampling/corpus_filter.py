"""Study corpus: the repositories that AI labelling, the results and the benchmark are based on.

Mining keeps every repository that is real Pharo code (about 1,600). Most of them are small,
inactive or personal, so labelling and benchmark seeds drawn from all of them describe GitHub
more than they describe Pharo software engineering. This module keeps the repositories that
show the practices of an engineered project and whose code can serve a benchmark that asks a
model to write and to fix Pharo code: redistributable, loadable, tested, original.

Every criterion is a separate measured threshold with its source in the literature
(docs/1-2-filtering.md). There is no combined score. Each repository gets every criterion it
fails, and the first one in ``CRITERIA`` order is its funnel reason. Only when more
repositories pass than ``max_repositories`` is a choice made, and it is made for diversity
(greedy coverage of domain, size and kind), never for popularity.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import random

from ..storage.database import Database
from .diversity_sampling import diversity
from .repository_filters import repository_profiles

STAGE = "filter-corpus"

# key, config key, what is required (format with the threshold), what the criterion protects, source
CRITERIA = [
    ("license", "require_license", "a licence that GitHub recognises",
     "Benchmark tasks keep the provenance of real code and are redistributed", "SWE-bench [6], CodeSearchNet [10], RepoCoder [8]"),
    ("contributors", "min_contributors", "at least {} contributors",
     "Community: not a personal repository", "Kalliamvakou et al. [1], Munaiah et al. [2]"),
    ("commits", "min_commits", "at least {} commits",
     "History: sustained development, not an upload", "Kalliamvakou et al. [1], Munaiah et al. [2]"),
    ("development_days", "min_development_days", "at least {} days between creation and last push",
     "History: developed over time", "Munaiah et al. [2], Multi-SWE-bench [7]"),
    ("last_push", "max_years_since_last_push", "last push at most {} years before the mining date",
     "Code written for a current Pharo, the one the benchmark runs in", "Kalliamvakou et al. [1], Multi-SWE-bench [7]"),
    ("baseline", "require_baseline", "a Metacello baseline (BaselineOf class)",
     "The project declares how it is loaded: it can be installed in an image", "SWE-bench [6], Multi-SWE-bench [7]"),
    ("tonel", "require_tonel", "source in Tonel format",
     "The source format of current Pharo; FileTree-only projects predate it", "Multi-SWE-bench [7]"),
    ("non_test_methods", "min_non_test_methods", "at least {} methods outside tests",
     "Enough production code to draw varied methods from", "Munaiah et al. [2], SWE-Bench++ [9]"),
    ("max_methods", "max_methods", "at most {} methods",
     "Image dumps and generated code, which would dominate method-level figures", "Lopes et al. [4]"),
    ("test_methods", "min_test_methods", "at least {} test methods",
     "Unit testing: behaviour is specified by tests, which code generation and bug fixing are scored with",
     "Munaiah et al. [2], SWE-bench [6], CoderEval [11]"),
    ("test_share", "min_test_method_share", "at least {:.0%} of the methods are tests",
     "Unit testing as a practice, not a token test class", "Munaiah et al. [2]"),
    ("duplication", "max_duplicated_method_share", "at most {:.0%} of the methods copied from another repository",
     "Original code: copies inflate counts and leak between tasks", "Lopes et al. [4], Allamanis [5]"),
    ("parse_failures", "max_parse_failure_share", "at most {:.0%} of the methods fail to parse",
     "Standard Pharo syntax: other dialects and broken files", "measurement validity"),
]


def filter_digest(cfg) -> str:
    """Fingerprint of the settings that decide the study corpus."""
    return hashlib.sha256(json.dumps(cfg["corpus_filter"], sort_keys=True, default=str).encode()).hexdigest()[:16]


def _date(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s[:10]) if s else None


def repository_metrics(db: Database, cfg) -> tuple[list[dict], str | None]:
    """One row per mined repository (latest snapshot with parsed methods), and the mining date
    that ``last_push`` is measured against, so the same data always gives the same corpus."""
    latest = "(SELECT MAX(snapshot_id) FROM repository_snapshots GROUP BY repo_id)"
    m = {r["repo_id"]: dict(r) for r in db.query(
        "SELECT repo_id, COUNT(*) methods, COALESCE(SUM(is_test), 0) test_methods,"
        " SUM(duplicate_of_method_id IS NOT NULL) duplicated_methods, SUM(parse_status != 'ok') parse_failures"
        f" FROM methods WHERE snapshot_id IN {latest} GROUP BY repo_id")}
    snap = {r["snapshot_id"]: r["repo_id"] for r in db.query(
        f"SELECT snapshot_id, repo_id FROM repository_snapshots WHERE snapshot_id IN {latest}")}
    for r in db.query("SELECT snapshot_id, COUNT(*) classes, SUM(name LIKE 'BaselineOf%') baselines FROM classes GROUP BY snapshot_id"):
        if snap.get(r["snapshot_id"]) in m:
            m[snap[r["snapshot_id"]]].update(classes=r["classes"], baselines=r["baselines"] or 0)
    for r in db.query("SELECT snapshot_id, SUM(format LIKE 'tonel%') tonel_files FROM source_files GROUP BY snapshot_id"):
        if snap.get(r["snapshot_id"]) in m:
            m[snap[r["snapshot_id"]]]["tonel_files"] = r["tonel_files"] or 0
    dates = {r["repo_id"]: r for r in db.query("SELECT repo_id, created_at, pushed_at, metadata_collected_at FROM repositories")}
    mined = max((d["metadata_collected_at"] or "" for d in dates.values()), default="") or max(
        (d["pushed_at"] or "" for d in dates.values()), default="")
    ref = _date(mined)
    out = []
    for p in repository_profiles(db, cfg):
        x = m.get(p["repo_id"])
        if x is None:
            continue
        d = dates[p["repo_id"]]
        created, pushed = _date(d["created_at"]), _date(d["pushed_at"])
        n = x["methods"]
        out.append({
            **{k: p[k] for k in ("repo_id", "full_name", "owner", "license", "stars", "forks", "contributors", "commit_count",
                                 "primary_domain", "size_category", "project_kind")},
            "pushed_at": d["pushed_at"], "created_at": d["created_at"],
            "development_days": (pushed - created).days if created and pushed else None,
            "years_since_last_push": round((ref - pushed).days / 365.25, 2) if ref and pushed else None,
            "classes": x.get("classes", 0), "baselines": x.get("baselines", 0), "tonel_files": x.get("tonel_files", 0),
            "methods": n, "test_methods": x["test_methods"], "non_test_methods": n - x["test_methods"],
            "test_method_share": round(x["test_methods"] / n, 4), "duplicated_method_share": round(x["duplicated_methods"] / n, 4),
            "parse_failure_share": round(x["parse_failures"] / n, 4),
        })
    return out, mined[:10] or None


def failed_criteria(m: dict, f: dict) -> list[str]:
    """Keys of every criterion the repository fails, in ``CRITERIA`` order. A criterion set to
    0/false is off. A value GitHub did not give (None) fails a minimum: the evidence is missing."""
    def below(value, key):
        t = f.get(key)
        return bool(t) and (value is None or value < t)

    def above(value, key):
        t = f.get(key)
        return t is not None and t is not False and value is not None and value > t

    checks = {
        "license": bool(f.get("require_license")) and m["license"] in (None, "", "NOASSERTION"),
        "contributors": below(m["contributors"], "min_contributors"),
        "commits": below(m["commit_count"], "min_commits"),
        "development_days": below(m["development_days"], "min_development_days"),
        "last_push": bool(f.get("max_years_since_last_push")) and (
            m["years_since_last_push"] is None or m["years_since_last_push"] > f["max_years_since_last_push"]),
        "baseline": bool(f.get("require_baseline")) and not m["baselines"],
        "tonel": bool(f.get("require_tonel")) and not m["tonel_files"],
        "non_test_methods": below(m["non_test_methods"], "min_non_test_methods"),
        "max_methods": bool(f.get("max_methods")) and m["methods"] > f["max_methods"],
        "test_methods": below(m["test_methods"], "min_test_methods"),
        "test_share": below(m["test_method_share"], "min_test_method_share"),
        "duplication": above(m["duplicated_method_share"], "max_duplicated_method_share"),
        "parse_failures": above(m["parse_failure_share"], "max_parse_failure_share"),
    }
    return [key for key, *_ in CRITERIA if checks[key]]


def requirement(key: str, f: dict) -> str:
    """The criterion as a sentence with its current threshold, or 'off'."""
    for k, cfg_key, text, *_ in CRITERIA:
        if k == key:
            t = f.get(cfg_key)
            share_limit = cfg_key in ("max_duplicated_method_share", "max_parse_failure_share")  # 0 is a real limit
            off = t is None or t is False or (not t and not share_limit)
            return "off" if off else text.format(t)
    raise KeyError(key)


def select(metrics: list[dict], f: dict) -> tuple[list[dict], list[dict], dict]:
    """Returns (decisions, funnel, info). A decision is the metrics row plus ``decision``
    (included | excluded | not_selected), ``reason`` and ``failed`` (all failed criteria)."""
    decisions, passing = [], []
    for m in sorted(metrics, key=lambda m: m["full_name"].lower()):
        failed = failed_criteria(m, f)
        row = {**m, "failed": failed}
        if failed:
            decisions.append({**row, "decision": "excluded", "reason": requirement(failed[0], f)})
        else:
            passing.append(row)
    cap = int(f.get("max_repositories") or 0)
    info = {"mined": len(metrics), "passing": len(passing), "max_repositories": cap, "capped": False}
    chosen, skipped = passing, []
    if cap and len(passing) > cap:
        chosen, left = diversity(passing, cap, list(f["diversity_attributes"]), random.Random(int(f["random_seed"])),
                                 int(f.get("max_per_owner") or 0))
        skipped = [p for p, _ in left]
        info["capped"] = True
    decisions += [{**p, "decision": "included", "reason": "passes every criterion"} for p in chosen]
    decisions += [{**p, "decision": "not_selected",
                   "reason": f"passes every criterion; not among the {cap} chosen for diversity"} for p in skipped]
    remaining, funnel = len(metrics), []
    for key, cfg_key, _, protects, source in CRITERIA:
        removed = sum(1 for d in decisions if d["failed"][:1] == [key])
        remaining -= removed
        funnel.append({"criterion": key, "requirement": requirement(key, f), "protects": protects, "source": source,
                       "fail_alone": sum(1 for d in decisions if key in d["failed"]), "removed": removed, "remaining": remaining})
    info["selected"] = len(chosen)
    return decisions, funnel, info


def study_repo_ids(db: Database) -> set[int] | None:
    """Repositories of the study corpus, or None when the corpus was never filtered (every
    analysed repository then counts, as before this step existed)."""
    run = db.scalar("SELECT run_id FROM analysis_runs WHERE stage=? AND status='completed' ORDER BY started_at DESC LIMIT 1",
                    (STAGE,))
    if not run:
        return None
    return {r[0] for r in db.query("SELECT repo_id FROM corpus_decisions WHERE run_id=? AND decision='included'", (run,))}


def write_report(path, cfg, funnel: list[dict], decisions: list[dict], info: dict, mined: str | None):
    f = cfg["corpus_filter"]
    inc = sorted((d for d in decisions if d["decision"] == "included"), key=lambda d: d["full_name"].lower())
    L = ["# Study corpus\n",
         f"{info['selected']} of {info['mined']} mined repositories form the study corpus: the repositories that AI labelling, "
         "the results and the benchmark are based on. Method and literature: `docs/1-2-filtering.md`. Thresholds: "
         "`corpus_filter` in `config.yaml`.\n",
         f"Mining date used for the last-push criterion: {mined or 'unknown'}.\n",
         "## Filtering funnel\n",
         "The criteria are applied in this order. *Fails alone* counts every mined repository that fails the criterion, "
         "whatever the others say; *removed* counts those for which it is the first one failed.\n",
         "| # | Requirement | Fails alone | Removed | Remaining | Source |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(funnel, 1):
        L.append(f"| {i} | {r['requirement']} | {r['fail_alone']} | {r['removed']} | {r['remaining']} | {r['source']} |")
    if info["capped"]:
        L.append(f"\n{info['passing']} repositories pass every criterion, more than the maximum of {info['max_repositories']}. "
                 f"{info['selected']} were chosen by greedy coverage of `{'`, `'.join(f['diversity_attributes'])}` "
                 f"(seed {f['random_seed']}, at most {f.get('max_per_owner') or 'any number'} per owner), following Nagappan et al. [3].\n")
    else:
        L.append(f"\n{info['passing']} repositories pass every criterion, within the maximum of {info['max_repositories']}: "
                 "all of them are kept and no further choice is made.\n")
    for title, key in (("Domains", "primary_domain"), ("Owners with most repositories", "owner")):
        counts = {}
        for d in inc:
            counts[d[key]] = counts.get(d[key], 0) + 1
        top = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:15]
        L += [f"## {title}\n", "| | Repositories |", "|---|---|"] + [f"| {k} | {v} |" for k, v in top] + [""]
    L += ["## Repositories of the study corpus\n",
          "| Repository | Domain | Licence | Contributors | Commits | Methods | Test methods | Copied | Last push |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d in inc:
        L.append(f"| {d['full_name']} | {d['primary_domain']} | {d['license']} | {d['contributors']} | {d['commit_count']} | "
                 f"{d['methods']} | {d['test_methods']} | {d['duplicated_method_share']:.0%} | {(d['pushed_at'] or '')[:10]} |")
    L.append("\nEvery mined repository, its measurements, its decision and every criterion it fails: "
             "`results/tables/corpus_filter.csv`.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
