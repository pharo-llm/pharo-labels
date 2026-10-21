"""Orchestrator: which phases to run, in which order, and what to skip.

    python -m pbr run                 every phase in order; a phase already done with the same inputs is skipped
    python -m pbr run 6               phase 6 only (always re-run; LLM answers come from the cache)
    python -m pbr run 3-7             phases 3 to 7
    python -m pbr run from 6          phases 6 to 12
    python -m pbr status              one line per phase: status, when, how long, key numbers, stale or not
    python -m pbr logs [6] [-n 60]    the log of the whole pipeline, or of one phase, cleanly
    python -m pbr reset 6             delete what phase 6 produced (human inputs in returned/ are kept)
    python -m pbr cost                LLM calls, tokens, time and cost per phase and model
    python -m pbr simulate-expert 7   a SIMULATED expert workbook to test the import path (never a human result)
"""
from __future__ import annotations

import argparse
import importlib
import re
import shutil
import sys

from .common import (BY_NUM, PHASES, PhaseWaiting, ROOT, STATUS_ICON, _hms, fingerprint, is_up_to_date, load_config,
                     phase_dirname, phase_state, read_jsonl, results_dir, running, show, write_summary, write_text)

MODULES = {1: "p01_literature", 2: "p02_corpus", 3: "p03_sampling", 4: "p04_discovery", 5: "p05_taxonomy",
           6: "p06_annotation", 7: "p07_validation", 8: "p08_profile", 9: "p09_coverage", 10: "p10_benchmark",
           11: "p11_verification", 12: "p12_evaluation"}


def parse_targets(args: list[str]) -> tuple[list[int], bool]:
    """(phases, force). No argument = all phases, unforced (skip what is up to date)."""
    if not args:
        return [p[0] for p in PHASES], False
    if args[0] == "from":
        start = int(args[1])
        return [p[0] for p in PHASES if p[0] >= start], True
    out = []
    for a in args:
        m = re.fullmatch(r"(\d+)-(\d+)", a)
        if m:
            out += list(range(int(m.group(1)), int(m.group(2)) + 1))
        elif a.isdigit():
            out.append(int(a))
        else:
            key = {p[1]: p[0] for p in PHASES}.get(a)
            if key is None:
                raise SystemExit(f"Unknown phase {a!r}: 1-12, a range like 3-7, 'from 6', or a name "
                                 f"({', '.join(p[1] for p in PHASES)})")
            out.append(key)
    bad = [n for n in out if n not in BY_NUM]
    if bad:
        raise SystemExit(f"No phase {bad}")
    return sorted(dict.fromkeys(out)), True


def phase_readme(cfg: dict, num: int):
    """results/NN_<phase>/README.md: what the folder holds and how it was made (the phase's documentation)."""
    mod = importlib.import_module(f"pbr.phases.{MODULES[num]}")
    write_text(results_dir(cfg) / phase_dirname(num) / "README.md",
               f"# Phase {num:02d} — {BY_NUM[num][2]}\n\n```\n{(mod.__doc__ or '').strip()}\n```\n\n"
               f"Status, metrics and the list of files: `_phase.json`. Log: `../logs/{phase_dirname(num)}.log` "
               f"(or `./run.sh logs {num}`).\n")


def run_phases(cfg: dict, phases: list[int], force: bool) -> int:
    for num in phases:
        title = BY_NUM[num][2]
        if not force and is_up_to_date(cfg, num):
            st = phase_state(cfg, num)
            print(f"{STATUS_ICON[st['status']]} {num:02d} {title}: up to date ({st['status']}, {st.get('finished', '')[:16]}) — skipped")
            phase_readme(cfg, num)
            continue
        missing = [u for u in BY_NUM[num][4] if phase_state(cfg, u).get("status") not in ("done", "provisional")]
        if missing:
            print(f"✗ {num:02d} {title}: needs phase(s) {missing} first — run ./run.sh {missing[0]} (or ./run.sh)")
            return 1
        stale_up = [u for u in BY_NUM[num][4] if not is_up_to_date(cfg, u)]
        if stale_up and not force:
            print(f"! {num:02d}: upstream phase(s) {stale_up} are stale")
        mod = importlib.import_module(f"pbr.phases.{MODULES[num]}")
        try:
            with running(cfg, num) as ph:
                mod.run(ph)
                phase_readme(cfg, num)
        except PhaseWaiting as e:
            print(f"\n⏸ Phase {num:02d} waits for a human decision:\n  {e}\nThen run ./run.sh again (it resumes here).")
            return 2
        except KeyboardInterrupt:
            print(f"\nInterrupted in phase {num:02d}. ./run.sh {num} resumes it (answers already obtained are cached).")
            return 130
        except Exception as e:   # noqa: BLE001 — reported, logged with traceback, run stops
            print(f"\n✗ Phase {num:02d} failed: {type(e).__name__}: {e}\n  log: results/logs/{phase_dirname(num)}.log"
                  f"\n  fix, then ./run.sh {num} (or ./run.sh from {num})")
            return 1
    return 0


def status(cfg: dict) -> int:
    rd = results_dir(cfg)
    print(f"Pharo benchmark research — profile {cfg['profile_name']} — {show(rd)}/")
    print(f"{'':2}{'phase':<46}{'status':<20}{'finished':<18}{'took':>8}  key numbers")
    for num, key, title, *_ in PHASES:
        st = phase_state(cfg, num)
        s = st.get("status")
        flag = ""
        if s in ("done", "provisional") and st.get("fingerprint") != fingerprint(cfg, num):
            flag = " (stale)"
        nums = ", ".join(f"{k}={v}" for k, v in (st.get("metrics") or {}).items()
                         if isinstance(v, (int, float, str)) and len(str(v)) < 24)
        took = _hms(st["duration_s"]) if st.get("duration_s") is not None else ""
        print(f"{STATUS_ICON.get(s, '?')} {f'{num:02d} {title}':<46}{(s or 'not run') + flag:<20}"
              f"{(st.get('finished') or '')[:16]:<18}{took:>8}  {nums[:90]}")
        if s == "waiting":
            print(f"     ⏸ {st.get('waiting_for')}")
        if s == "failed":
            print(f"     ✗ {st.get('error')}")
    print("\n✓ done  ◐ provisional (will change once humans answer)  ⏸ waiting  ✗ failed  · not run  (stale) = inputs changed")
    print("Summary with figures: results/SUMMARY.md   Logs: ./run.sh logs [phase]")
    return 0


def logs(cfg: dict, target: str | None, n: int) -> int:
    d = results_dir(cfg) / "logs"
    if target:
        num, _ = parse_targets([target])
        f = d / f"{phase_dirname(num[0])}.log"
    else:
        f = d / "pipeline.log"
    if not f.exists():
        print(f"No log yet: {show(f)}")
        return 1
    lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
    # only the last run of the phase (a run starts with a line of '=')
    if target:
        starts = [i for i, l in enumerate(lines) if l.rstrip().endswith("=" * 78)]
        if starts:
            lines = lines[starts[-1]:]
    colour = sys.stdout.isatty()
    for line in lines[-n:]:
        if colour and "| ERROR" in line:
            line = f"\033[31m{line}\033[0m"
        elif colour and "| WARNING" in line:
            line = f"\033[33m{line}\033[0m"
        elif colour and ("PHASE " in line):
            line = f"\033[1m{line}\033[0m"
        print(line)
    print(f"\n({show(f)}; {len(lines)} lines shown up to {n}; Traceback details are only in the file)")
    return 0


def reset(cfg: dict, target: str) -> int:
    nums, _ = parse_targets([target])
    for num in nums:
        d = results_dir(cfg) / phase_dirname(num)
        if not d.exists():
            continue
        keep = set(BY_NUM[num][5])
        for child in d.iterdir():
            rel = child.name
            if any(k == rel or k.startswith(rel + "/") for k in keep):
                if child.is_dir():   # keep returned/ folders, clear the rest of their parent
                    for sub in child.iterdir():
                        if not any(k == f"{rel}/{sub.name}" for k in keep):
                            shutil.rmtree(sub) if sub.is_dir() else sub.unlink()
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        print(f"reset {num:02d} {BY_NUM[num][2]} (human inputs kept: {', '.join(keep) or 'none'})")
    write_summary(cfg)
    return 0


def cost(cfg: dict) -> int:
    rows = read_jsonl(results_dir(cfg) / "logs" / "llm_calls.jsonl")
    if not rows:
        print("No LLM calls yet.")
        return 0
    agg = {}
    for r in rows:
        k = (r.get("phase", ""), r["model"])
        a = agg.setdefault(k, [0, 0, 0, 0.0, 0.0, 0.0, 0])
        a[0] += 1
        a[1] += int(r.get("input_tokens") or 0)
        a[2] += int(r.get("output_tokens") or 0)
        a[3] += float(r.get("latency_s") or 0)
        a[4] += float(r.get("cost_usd") or 0)
        a[5] += float(r.get("reference_cost_usd") or 0)
        a[6] += 0 if r.get("valid") else 1
    ref = cfg["llm"].get("reference_price", {}).get("name", "reference")
    print(f"{'phase':<14}{'model':<16}{'calls':>7}{'invalid':>8}{'in tok':>11}{'out tok':>10}{'time':>9}{'cost $':>9}{('on ' + ref)[:18]:>20}")
    for (p, m), a in sorted(agg.items()):
        print(f"{p:<14}{m:<16}{a[0]:>7}{a[6]:>8}{a[1]:>11,}{a[2]:>10,}{_hms(a[3]):>9}{a[4]:>9.2f}{a[5]:>20.2f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pbr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="run")
    ap.add_argument("args", nargs="*")
    ap.add_argument("-n", type=int, default=80, help="lines of log to show")
    ap.add_argument("--profile", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(profile=a.profile)
    results_dir(cfg).mkdir(parents=True, exist_ok=True)
    if a.command == "run":
        phases, force = parse_targets(a.args)
        code = run_phases(cfg, phases, force)
        write_summary(cfg)
        return code
    if a.command == "status":
        return status(cfg)
    if a.command == "logs":
        return logs(cfg, a.args[0] if a.args else None, a.n)
    if a.command == "reset":
        if not a.args:
            raise SystemExit("reset <phase>")
        return reset(cfg, a.args[0])
    if a.command == "cost":
        return cost(cfg)
    if a.command == "simulate-expert":
        from .phases import simulate
        return simulate.main(cfg, a.args)
    raise SystemExit(f"Unknown command {a.command!r}")
