"""Print the key numbers of a run in a few readable lines (used by job.fir.sh after each step).

    python scripts/job_summary.py RESULTS_DIR [corpus|labelling|results|benchmark|all]

Reads only files the steps write; prints nothing for a step that has not produced them yet.
"""
import csv
import json
import sys
from pathlib import Path


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def rows(path: Path) -> list[dict]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def scalars(d: dict, indent: str = "    ", depth: int = 0) -> None:
    """Numbers and short strings of a nested dict, one per line (agreement metrics have no fixed shape)."""
    for k, v in d.items():
        if isinstance(v, dict) and depth < 1:
            print(f"{indent}{k}:")
            scalars(v, indent + "  ", depth + 1)
        elif isinstance(v, float):
            print(f"{indent}{k}: {v:.3f}")
        elif isinstance(v, (int, str)) and len(str(v)) < 80:
            print(f"{indent}{k}: {v}")


def corpus(r: Path) -> None:
    c = load(r / "data/processed/corpus.json")
    if not c:
        return
    print(f"  Study corpus: {len(c.get('selected', []))} repositories kept (mining date {c.get('mining_date', '?')})")
    funnel = [f for f in c.get("funnel") or [] if isinstance(f, dict)]
    if funnel:
        print(f"    {'criterion':<18} {'removed':>7} {'remaining':>9}   requirement")
    for f in funnel:
        print(f"    {f.get('criterion', '?'):<18} {f.get('removed', '-'):>7} {f.get('remaining', '-'):>9}   "
              f"{f.get('requirement', '')}")
    sample = rows(r / "validation/annotation_sample.csv")
    if sample:
        repos = {s.get("repository") or s.get("full_name") or s.get("repo") for s in sample}
        print(f"  Methods to label: {len(sample)} from {len(repos - {None})} repositories "
              f"(results/validation/annotation_sample.csv)")


def labelling(r: Path) -> None:
    m = load(r / "validation/llm/run_manifest.json")
    if not m:
        return
    print(f"  AI labelling: {m.get('items')} methods, aggregation {m.get('aggregation')}, "
          f"evidence policy {m.get('evidence_policy')}")
    print(f"    final status: {m.get('status_counts')}")
    for name, st in (m.get("per_model") or {}).items():
        short = {k: v for k, v in st.items() if isinstance(v, (int, float, str))} if isinstance(st, dict) else st
        print(f"    {name}: {short}")
    if isinstance(m.get("agreement"), dict):
        print("    agreement between the models:")
        scalars(m["agreement"], "      ")
    fails = rows(r / "validation/llm/failures.csv")
    if fails:
        print(f"    failures: {len(fails)} answers could not be used (results/validation/llm/failures.csv)")


def results(r: Path) -> None:
    reports = sorted(p.name for p in (r / "reports").glob("*.md"))
    figures = list((r / "figures").glob("*.png"))
    tables = list((r / "tables").glob("*.csv"))
    if reports:
        print(f"  Reports ({len(reports)}): {', '.join(reports)}")
        print(f"  Figures: {len(figures)}   Tables: {len(tables)}   Start with results/reports/ecosystem_summary.md")


def benchmark(r: Path) -> None:
    m = load(r / "benchmark/manifest.json")
    if not m:
        return
    print(f"  Benchmark: {m.get('accepted')} tasks accepted, {m.get('rejected')} rejected, "
          f"of {m.get('candidates')} candidates")
    for f in m.get("funnel") or []:
        print(f"    {f.get('gate', '?'):<22} rejected {f.get('rejected', 0):>4}   remaining {f.get('remaining', '?')}")
    if m.get("mutation_score"):
        print(f"    mutation score: {m['mutation_score']}")
    if m.get("tests_per_task") is not None:
        print(f"    tests per task: {m['tests_per_task']}   repaired: {m.get('repaired_tasks')}   "
              f"with added tests: {m.get('tasks_with_added_tests')}")


def main() -> None:
    r = Path(sys.argv[1])
    what = sys.argv[2] if len(sys.argv) > 2 else "all"
    for name, fn in (("corpus", corpus), ("labelling", labelling), ("results", results), ("benchmark", benchmark)):
        if what in (name, "all"):
            try:
                fn(r)
            except Exception as e:  # a summary must never stop the job
                print(f"  ({name} summary unavailable: {e})")


if __name__ == "__main__":
    main()
