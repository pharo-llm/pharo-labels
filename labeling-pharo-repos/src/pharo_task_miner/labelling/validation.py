"""Step 5: the experts' answers, read back.

Each expert classifies the methods of ``method_review.csv`` blind (the models' answers are not
shown) and reviews the codebook. Their files go in ``expert_validation/returned/<expert>/``. After
the experts have discussed the methods they disagree on (``to_discuss.csv``), the agreed answers
go in ``returned/consensus/`` (same file, only the methods discussed are needed).

* **Reference label** of a method: the experts' common answer; ``consensus`` when they differ;
  methods still disputed are left out (and counted).
* **Experts' agreement**: Krippendorff's alpha and pairwise Cohen's kappa: the human baseline.
  A model can stand in for a human rater when its agreement with the reference is close to the
  experts' agreement with each other (Ahmed et al., MSR 2025).
* **Accuracy** of each model and of the majority label: stratified estimator over the agreement
  levels (weights = methods in the level / methods classified), 95% interval; and the accuracy
  inside each level (does agreement between models predict correctness?).
* **Precision and recall** of the majority label per category, weighted by the stratum weights.
* **Share of each category in the corpus**: the models' share on the classified sample corrected
  with the experts' answers (prediction-powered inference, ``agreement.ppi_share``).
* **Codebook decisions** (keep / rename / merge / definition) by majority of the experts, applied
  to the models' and the experts' labels before the numbers above; ties go to ``to_discuss.csv``.

Output: ``results/labelling/validated/`` (validation.json, final_taxonomy.csv/.md,
reference_labels.csv, to_discuss.csv).
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from ..storage.provenance import write_json
from . import agreement as ag
from .classify import OTHER, load_codebook
from .export import _csv

FILES = ("method_review.csv", "codebook_review.csv")
CONSENSUS = "consensus"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]


def _yes(v: str) -> bool | None:
    v = (v or "").strip().lower()
    if v in ("yes", "y", "true", "1", "oui", "x"):
        return True
    if v in ("no", "n", "false", "0", "non"):
        return False
    return None


def _norm(name: str) -> str:
    return " ".join((name or "").split()).casefold()


def returned(ev: Path) -> dict[str, dict[str, list[dict]]]:
    """expert name -> file -> rows, for every expert folder with at least one answer."""
    out = {}
    back = ev / "returned"
    for d in sorted(p for p in back.iterdir() if p.is_dir()) if back.exists() else []:
        files = {f: _read(d / f) for f in FILES}
        if any(r.get("expert_category") for r in files["method_review.csv"]) or \
                any(_yes(r.get("keep_yes_no")) is not None for r in files["codebook_review.csv"]):
            out[d.name] = files
    return out


def _majority(values: list) -> tuple[object, bool]:
    votes = Counter(v for v in values if v not in (None, ""))
    if not votes:
        return None, True
    (top, k), *rest = votes.most_common()
    return top, not rest or k > rest[0][1]


def codebook_decisions(experts: dict, names: list[str]) -> tuple[dict, list[dict]]:
    rows = defaultdict(list)
    for e, files in experts.items():
        for r in files["codebook_review.csv"]:
            if r.get("category"):
                rows[r["category"]].append((e, r))
    out, disputed = {}, []
    for cat, answers in rows.items():
        d = {"definition_fix": " / ".join(dict.fromkeys(r["definition_fix"] for _, r in answers if r.get("definition_fix")))}
        for field, col, conv in (("keep", "keep_yes_no", _yes), ("rename", "rename_to", str.strip),
                                 ("merge", "merge_into", str.strip)):
            top, strict = _majority([conv(r.get(col, "")) for _, r in answers])
            if not strict:
                disputed.append({"what": f"codebook: {field}", "item": cat,
                                 "answers": "; ".join(f"{e}: {r.get(col, '')}" for e, r in answers)})
                top = None
            d[field] = top
        out[cat] = d
    return out, disputed


def resolver(decisions: dict, names: list[str]):
    """codebook category -> final category (merges followed, renamed; dropped -> Other)."""
    by_norm = {_norm(n): n for n in names}

    def final(name: str | None) -> str | None:
        if not name:
            return None
        cur, seen = by_norm.get(_norm(name), name), set()
        while decisions.get(cur, {}).get("merge") and cur not in seen:
            seen.add(cur)
            target = decisions[cur]["merge"]
            cur = by_norm.get(_norm(target), target)
        d = decisions.get(cur, {})
        if d.get("keep") is False and not d.get("merge"):
            return OTHER
        return d.get("rename") or cur
    return final


def expert_label(r: dict, names: list[str]) -> str | None:
    """The codebook category an expert wrote (any other name counts as Other)."""
    v = r.get("expert_category", "")
    if not v:
        return None
    by_norm = {_norm(n): n for n in names}
    return by_norm.get(_norm(v), OTHER)


def validate(out: Path, log) -> dict | None:
    ev = out / "expert_validation"
    cdir = out / "classify"
    if (cdir / "NOT_RELIABLE.md").exists() or not (cdir / "gold_selection.csv").exists():
        log("No expert files were produced (step 4 not run, or the codebook was not reliable): nothing to validate.")
        return None
    experts = returned(ev)
    consensus = experts.pop(CONSENSUS, None)
    if not experts:
        log(f"No expert answers yet. Put each expert's filled files in {ev / 'returned' / '<expert name>'}/ "
            f"({', '.join(FILES)}) and run this step again.")
        return None
    plan = json.loads((out / "plan.json").read_text())
    conf, N = plan["sample_size"]["confidence"], plan["population"]["methods"]
    res4 = json.loads((cdir / "agreement.json").read_text())
    codebook, _ = load_codebook(out)
    names = [c["category"] for c in codebook]
    models = res4["models"]
    sel = {int(r["method_id"]): r for r in _read(cdir / "gold_selection.csv")}
    allrows = _read(cdir / "labels.csv")
    n = len(allrows)
    log(f"Expert answers from: {', '.join(experts)}" + (" (+ consensus)" if consensus else ""))

    decisions, disputed = codebook_decisions(experts, names)
    final = resolver(decisions, names)

    # ------------------------------------------------------------------ reference labels
    answers = defaultdict(dict)
    for e, files in experts.items():
        for r in files["method_review.csv"]:
            if r.get("method_id") and int(r["method_id"]) in sel:
                lab = expert_label(r, names)
                if lab:
                    answers[int(r["method_id"])][e] = final(lab)
    agreed = {int(r["method_id"]): final(expert_label(r, names)) for r in (consensus or {}).get("method_review.csv", [])
              if r.get("method_id") and expert_label(r, names)}
    ref, how = {}, Counter()
    for m, by in answers.items():
        top, strict = _majority(list(by.values()))
        if len(set(by.values())) == 1 or (strict and len(by) > 2):
            ref[m] = top
            how["experts agree" if len(set(by.values())) == 1 else "expert majority"] += 1
        elif m in agreed:
            ref[m] = agreed[m]
            how["consensus after discussion"] += 1
        else:
            disputed.append({"what": "method", "item": str(m),
                             "answers": "; ".join(f"{e}: {v}" for e, v in sorted(by.items()))})
            how["still disputed (left out)"] += 1

    enames = sorted(experts)
    units = [[answers[m].get(e) for e in enames] for m in sel]
    expert_pairs = [{"experts": [a, b], "cohen_kappa": ag.cohen_kappa([u[i] for u in units], [u[j] for u in units]),
                     "methods": sum(1 for u in units if u[i] and u[j])}
                    for i, a in enumerate(enames) for j, b in enumerate(enames) if i < j]
    human_alpha = ag.krippendorff_alpha(units) if len(enames) > 1 else None

    # ------------------------------------------------------------------ accuracy, stratified
    strata = {}
    for r in sel.values():
        strata[r["stratum"]] = {"N": int(r["stratum_methods"]), "m": int(r["stratum_checked"])}
    labelled = {m: {mod: final(sel[m].get(mod) or None) for mod in models} | {"majority": final(sel[m].get("majority") or None)}
                for m in ref}

    def accuracy(who: str) -> dict:
        vals = defaultdict(list)
        for m in ref:
            vals[sel[m]["stratum"]].append(1.0 if labelled[m][who] == ref[m] else 0.0)
        res = ag.stratified_mean(strata, vals, conf)
        res["by_level"] = {h: {"correct": int(sum(v)), "checked": len(v), "accuracy": round(sum(v) / len(v), 3)}
                           for h, v in sorted(vals.items())}
        return res

    acc = {who: accuracy(who) for who in models + ["majority"]}
    model_vs_ref = {mod: ag.krippendorff_alpha([[labelled[m][mod], ref[m]] for m in ref]) for mod in models + ["majority"]}

    # ------------------------------------------------------------------ per category: precision, recall, PPI share
    weight = {m: strata[sel[m]["stratum"]]["N"] / strata[sel[m]["stratum"]]["m"] for m in ref}
    final_names = list(dict.fromkeys(final(c) for c in names))
    full_major = Counter(final(r["majority"] or None) for r in allrows)
    table = []
    for c in final_names:
        tp = sum(weight[m] for m in ref if labelled[m]["majority"] == c and ref[m] == c)
        pp = sum(weight[m] for m in ref if labelled[m]["majority"] == c)
        ap = sum(weight[m] for m in ref if ref[m] == c)
        diffs = defaultdict(list)
        for m in ref:
            diffs[sel[m]["stratum"]].append((ref[m] == c) - (labelled[m]["majority"] == c))
        share = ag.ppi_share(full_major[c] / n, n, strata, diffs, conf, N)
        table.append({"category": c, "definition": "", **share,
                      "estimated_methods_in_corpus": round(share["share"] * N),
                      "precision": round(tp / pp, 3) if pp else None, "recall": round(tp / ap, 3) if ap else None,
                      "checked_by_experts": sum(1 for m in ref if ref[m] == c)})
    defs = {final(c["category"]): c["definition"] for c in codebook}
    for cat, d in decisions.items():
        if d.get("definition_fix") and final(cat) == (d.get("rename") or cat):
            defs[final(cat)] = d["definition_fix"]
    for t in table:
        t["definition"] = defs.get(t["category"], "")
    table.sort(key=lambda t: -t["share"])

    vdir = out / "validated"
    _csv(vdir / "final_taxonomy.csv", table, list(table[0]) if table else ["category"])
    _csv(vdir / "to_discuss.csv", disputed, ["what", "item", "answers"])
    _csv(vdir / "reference_labels.csv", [{"method_id": m, "stratum": sel[m]["stratum"], "reference": ref[m],
                                          **{k: v or "" for k, v in labelled[m].items()},
                                          **{f"expert_{e}": answers[m].get(e, "") for e in enames}} for m in sorted(ref)],
         ["method_id", "stratum", "reference", *models, "majority", *[f"expert_{e}" for e in enames]])
    md = ["# Final task taxonomy (after expert validation)\n",
          f"Shares of the corpus ({N:,} methods), corrected with {len(ref)} expert-checked methods "
          "(prediction-powered inference), 95% intervals.\n",
          "| Category | Definition | Share [95% CI] | ≈ methods in corpus | Precision | Recall |", "|---|---|---|---:|---:|---:|"]
    for t in table:
        md.append(f"| {t['category']} | {t['definition']} | {t['share']:.1%} [{t['ci_low']:.1%}, {t['ci_high']:.1%}] | "
                  f"{t['estimated_methods_in_corpus']:,} | {t['precision'] if t['precision'] is not None else '–'} | "
                  f"{t['recall'] if t['recall'] is not None else '–'} |")
    (vdir / "final_taxonomy.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # how often each expert changed the label the models proposed (method_review.csv, model_label):
    # with a pre-filled label, the experts lean toward it (Fort & Sagot 2010; Berzak et al. 2016)
    changed = {}
    for e, files in experts.items():
        rows = [r for r in files["method_review.csv"] if r.get("model_label") and r.get("expert_category")]
        if rows:
            changed[e] = {"methods": len(rows), "changed": sum(_norm(r["expert_category"]) != _norm(r["model_label"]) for r in rows)}
    result = {"experts": enames, "changed_model_label": changed, "consensus_file": bool(consensus), "reference": dict(how), "methods_with_reference": len(ref),
              "human_alpha": human_alpha, "expert_pairs": expert_pairs, "model_vs_reference_alpha": model_vs_ref,
              "accuracy": acc, "models": models, "strata": strata,
              "decisions": {"dropped": sum(1 for d in decisions.values() if d.get("keep") is False and not d.get("merge")),
                            "renamed": sum(1 for d in decisions.values() if d.get("rename")),
                            "merged": sum(1 for d in decisions.values() if d.get("merge")),
                            "definitions_fixed": sum(1 for d in decisions.values() if d.get("definition_fix")),
                            "to_discuss": len(disputed)},
              "final_taxonomy": table}
    write_json(vdir / "validation.json", result)
    a = acc["majority"]
    log(f"Accuracy of the majority label: {a['estimate']:.0%} [{a['ci_low']:.0%}, {a['ci_high']:.0%}] "
        f"({len(ref)} methods with a reference)")
    for mod in models:
        log(f"  {mod}: {acc[mod]['estimate']:.0%} [{acc[mod]['ci_low']:.0%}, {acc[mod]['ci_high']:.0%}]")
    if human_alpha is not None:
        log(f"Experts' agreement: Krippendorff's alpha {human_alpha}")
    for e, c in changed.items():
        log(f"  {e} changed the models' label of {c['changed']} of {c['methods']} methods")
    log(f"Final taxonomy: {len(table)} categories -> {vdir / 'final_taxonomy.md'}"
        + (f"; {len(disputed)} items to discuss: {vdir / 'to_discuss.csv'}" if disputed else ""))
    return result
