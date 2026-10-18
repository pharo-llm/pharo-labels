"""The labelling run: initial sample, saturation loop, check model.

Everything is written to ``results/labelling/`` as it happens, so an interrupted run resumes
where it stopped (labelled methods are never sent to the model again):

* ``plan.json``          population, sample-size calculation, model, settings
* ``rounds/round_NNN.json``  the methods drawn in each round (round 0 = the initial sample)
* ``labels.jsonl``       one line per labelled method: summary, raw task, category, model output
* ``curve.csv``          the saturation curve, one row per round
* ``check.jsonl``        the check model's labels of the review subset
* ``codebook.json``      every proposed category: support, variants, first and admission round

Every answer records a hash of its prompt. When a setting changes the prompt of a few methods only
(``max_method_chars``), the resumed run asks the model again for those methods and keeps the rest.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from pathlib import Path

from ..storage.provenance import now_iso, write_json
from . import coding, sampling


def _settings_digest(s: dict) -> str:
    """The settings that change the model's answers or the methods drawn. Codebook settings
    (min_support, merge thresholds, embedding model) are not in it: each method is coded on its
    own, so changing them replays the stored answers instead of calling the model again. Nor is
    ``max_method_chars``: it changes the prompt of very long methods only, which are labelled
    again on their own (``Run.refresh``)."""
    keys = ("model", "random_seed", "sample_size", "batch_size", "include_tests", "max_method_lines", "max_tokens",
            "prompt_version", "corpus_run_id")
    return hashlib.sha256(json.dumps({k: s.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()[:16]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass  # a line cut by an interruption
    return out


def _append(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rewrite(path: Path, rows: list[dict]):
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def _sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


class Run:
    def __init__(self, out: Path, settings: dict, pop: dict[int, list[int]], records, log, embed=None):
        self.out, self.s, self.pop, self.records, self.log = out, settings, pop, records, log
        self.embed = embed
        self._replay(None)
        self.curve: list[dict] = []

    def _replay(self, max_round: int | None):
        """Rebuild the codebook from the stored answers (same order, end-of-round admission);
        ``max_round``: ignore the answers of later rounds (drawn by an earlier, longer run)."""
        s = self.s
        self.codebook = coding.Codebook(s["min_support"], s["merge_similarity"], embed=self.embed,
                                        semantic_threshold=s.get("semantic_threshold", 0.78),
                                        min_share=s.get("min_share", 0.0))
        self.labels: dict[int, dict] = {}
        left = {}
        for row in _read_jsonl(self.out / "labels.jsonl"):
            r = row["round"]
            if row["method_id"] in self.labels or (max_round is not None and r > max_round):
                continue
            self.labels[row["method_id"]] = row
            self.codebook.observe(row["raw_task"], r)
            if r not in left:
                left[r] = set(self._round_ids(r) or [])
            left[r].discard(row["method_id"])
            if not left[r]:                                  # the round is complete: end-of-round admission
                self.codebook.admit(r, len(self.labels))

    # ------------------------------------------------------------------------------- prompts
    def prompt(self, method: dict) -> str:
        return coding.build_prompt(method, self.s["max_method_lines"], self.s.get("max_method_chars"))

    def stale(self, rows) -> list[tuple[dict, dict]]:
        """(stored answer, method) for the answers whose prompt is not the one the settings build now.
        An answer stored without a prompt hash was made before ``max_method_chars`` existed."""
        rows = list(rows)
        recs = self.records([r["method_id"] for r in rows])
        out = []
        for r in rows:
            m = recs.get(r["method_id"])
            if m is None:
                continue
            old = r.get("prompt_sha") or _sha(coding.build_prompt(m, self.s["max_method_lines"]))
            if old != _sha(self.prompt(m)):
                out.append((r, m))
        return out

    def _answer(self, model, method: dict) -> dict:
        prompt = self.prompt(method)
        t = time.monotonic()
        raw = model.complete(prompt, self.s["max_tokens"], coding.STOP)
        summary, task = coding.parse_answer(raw)
        return {"summary": summary, "raw_task": task, "model": model.name, "prompt_version": coding.PROMPT_VERSION,
                "prompt_sha": _sha(prompt), "seconds": round(time.monotonic() - t, 3), "output": raw}

    def refresh(self, model) -> int:
        """Ask the model again for the stored answers whose prompt changed, in place, so the rounds
        and the order in which the codebook sees the answers stay the same."""
        stale = self.stale(self.labels.values())
        if not stale:
            return 0
        self.log(f"  {len(stale)} stored answer(s) were made with another prompt: labelling them again.")
        new = {}
        for r, m in stale:
            ans = self._answer(model, m)
            task = ans["raw_task"]
            near = task and (self.codebook.match(task) or self.codebook.semantic_match(task)[0])
            new[r["method_id"]] = {**r, **ans, "category_at_labelling": near or task}
        _rewrite(self.out / "labels.jsonl", [new.get(r["method_id"], r) for r in _read_jsonl(self.out / "labels.jsonl")])
        self._replay(None)
        return len(stale)

    # ------------------------------------------------------------------------------- rounds
    def _round_ids(self, round_no: int) -> list[int] | None:
        p = self.out / "rounds" / f"round_{round_no:03d}.json"
        return json.loads(p.read_text())["method_ids"] if p.exists() else None

    def drawn_before(self, round_no: int) -> set[int]:
        out = set()
        for r in range(round_no):
            p = self.out / "rounds" / f"round_{r:03d}.json"
            if p.exists():
                out |= set(json.loads(p.read_text())["method_ids"])
        return out

    def draw(self, round_no: int) -> list[int]:
        p = self.out / "rounds" / f"round_{round_no:03d}.json"
        if p.exists():
            return json.loads(p.read_text())["method_ids"]
        size = self.s["sample_size"] if round_no == 0 else self.s["batch_size"]
        ids = sampling.draw(self.pop, size, self.s["random_seed"] + round_no, self.drawn_before(round_no))
        write_json(p, {"round": round_no, "size": len(ids), "drawn_at": now_iso(), "method_ids": ids})
        return ids

    def label_round(self, model, round_no: int, ids: list[int]) -> dict:
        todo = [m for m in ids if m not in self.labels]
        recs = self.records(todo)
        new_names, t0 = 0, time.monotonic()
        for i, mid in enumerate(todo, 1):
            m = recs[mid]
            ans = self._answer(model, m)
            task = ans["raw_task"]
            known = task is not None and self.codebook.match(task) is not None
            canon = self.codebook.observe(task, round_no)
            new_names += task is not None and not known
            row = {"method_id": mid, "round": round_no, "repository": m["full_name"], "signature": m["signature"],
                   "is_test": m["is_test"], "category_at_labelling": canon, **ans}
            self.labels[mid] = row
            _append(self.out / "labels.jsonl", row)
            if i % 25 == 0 or i == len(todo):
                self.log(f"    round {round_no}: {i}/{len(todo)} labelled, {len(self.codebook.taxonomy())} categories")
        if todo:
            self.codebook.admit(round_no, len(self.labels))
        return {"labelled_now": len(todo), "seconds": round(time.monotonic() - t0, 1), "new_names": new_names}

    def curve_row(self, round_no: int, ids: list[int], res: dict) -> dict:
        rows = [self.labels[m] for m in ids if m in self.labels]
        cb = self.codebook
        added = [c for c, v in cb.cats.items() if v["admitted_round"] == round_no]
        return {"round": round_no, "methods_in_round": len(ids),
                "methods_total": sum(1 for r in self.labels.values() if r["round"] <= round_no),
                "new_categories": len(added), "new_category_names": "; ".join(added),
                "categories_total": len(cb.taxonomy(round_no)),
                "candidates_total": sum(1 for n, v in cb.cats.items() if v["first_round"] is not None
                                        and v["first_round"] <= round_no and not cb.in_taxonomy(n, round_no)),
                "unparsable": sum(1 for r in rows if not r["raw_task"]),
                "new_names_proposed": sum(1 for v in cb.cats.values() if not v["seed"] and v["first_round"] == round_no),
                "seconds": res.get("seconds")}

    # ------------------------------------------------------------------------------- loop
    def run(self, model) -> dict:
        self.refresh(model)
        patience, quiet, stop = self.s["patience"], 0, "max_rounds"
        for round_no in range(0, self.s["max_rounds"] + 1):
            ids = self.draw(round_no)
            if not ids:
                stop = "population_exhausted"
                break
            what = "initial sample" if round_no == 0 else "saturation round"
            self.log(f"  {what} {round_no}: {len(ids)} methods")
            res = self.label_round(model, round_no, ids)
            row = self.curve_row(round_no, ids, res)
            self.curve.append(row)
            self.log(f"  round {round_no}: {row['new_categories']} new categor{'y' if row['new_categories'] == 1 else 'ies'}"
                     f"{' (' + row['new_category_names'] + ')' if row['new_categories'] else ''}; "
                     f"{row['categories_total']} in total, {row['candidates_total']} candidates")
            if round_no == 0:
                continue
            quiet = quiet + 1 if row["new_categories"] == 0 else 0
            if quiet >= patience:
                stop = "saturated"
                break
        last = self.curve[-1]["round"]
        if any(r["round"] > last for r in self.labels.values()):
            self.log(f"  answers of rounds after {last} (from an earlier, longer run) are kept on disk but not used.")
            self._replay(last)
        self._write_curve()
        self._write_codebook()
        return {"stop_reason": stop, "rounds": len(self.curve) - 1, "methods_labelled": len(self.labels),
                "categories": len(self.codebook.taxonomy())}

    def _write_curve(self):
        p = self.out / "curve.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.curve[0]))
            w.writeheader()
            w.writerows(self.curve)

    def _write_codebook(self):
        write_json(self.out / "codebook.json", {
            "min_support": self.codebook.min_support, "min_share": self.codebook.min_share,
            "merge_similarity": self.codebook.similarity,
            "semantic_threshold": self.codebook.semantic_threshold if self.codebook.embed else None,
            "prompt_version": coding.PROMPT_VERSION, "seed_categories": coding.SEED_CATEGORIES,
            "categories": [{**c, "variants": sorted(c["variants"]), "in_taxonomy": self.codebook.in_taxonomy(n)}
                           for n, c in self.codebook.cats.items()]})

    # ------------------------------------------------------------------------------- check model
    def check(self, model, size: int) -> list[dict]:
        """A second model codes a random subset independently, with the same prompt; both answers
        go through the same codebook, so agreement means the same final category."""
        path = self.out / "check.jsonl"
        rows = _read_jsonl(path)
        done = {r["method_id"]: r for r in rows if r.get("model") == model.name}
        stale = {r["method_id"] for r, _ in self.stale(done.values())}
        if stale:                                     # made with another prompt: coded again below
            _rewrite(path, [r for r in rows if not (r.get("model") == model.name and r["method_id"] in stale)])
            done = {k: v for k, v in done.items() if k not in stale}
        ids = self.review_ids(size)
        recs = self.records([m for m in ids if m not in done])
        for i, mid in enumerate([m for m in ids if m not in done], 1):
            ans = self._answer(model, recs[mid])
            row = {"method_id": mid, **{k: ans[k] for k in ("model", "summary", "raw_task", "prompt_sha", "output")}}
            done[mid] = row
            _append(path, row)
            if i % 20 == 0:
                self.log(f"    check: {i} labelled")
        for r in done.values():
            r["category"] = self.codebook.final(r["raw_task"])
        return [done[m] for m in ids]

    def review_ids(self, size: int) -> list[int]:
        ids = sorted(self.labels)
        rng = random.Random(self.s["random_seed"])
        return sorted(rng.sample(ids, min(size, len(ids))))


def prepare(out: Path, settings: dict, log) -> None:
    """Start fresh (moving old results aside) when the settings that shape the labels changed."""
    plan = out / "plan.json"
    digest = _settings_digest(settings)
    if plan.exists():
        old = json.loads(plan.read_text()).get("settings_digest")
        if old == digest:
            log(f"Resuming the labelling in {out} (same settings).")
            return
        dest = out.with_name(f"{out.name}.before-{now_iso()[:19].replace(':', '').replace('-', '')}")
        out.rename(dest)
        log(f"The labelling settings changed: previous results moved to {dest}.")
    out.mkdir(parents=True, exist_ok=True)
    settings["settings_digest"] = digest
