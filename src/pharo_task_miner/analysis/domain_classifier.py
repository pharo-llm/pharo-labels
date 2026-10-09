"""Module 4 — multi-label application-domain classification (RQ1).

Level 1 (deterministic): keyword evidence in description/README, topics,
BaselineOf dependencies, package names and referenced framework classes.
Level 2 (semantic, optional): see ``semantic_classifier.classify_repository_domain``.
Level 3 (human): labels are exported for review with ``review_status=pending``.

Scores are uncalibrated weight sums and are labelled as such. The repository
name is used only when no description, README or topics exist.
"""
from __future__ import annotations

from collections import Counter
import math
import re

W_TOPIC, W_DESC, W_README, W_DEP, W_FRAMEWORK, W_PACKAGE, W_NAME = 2.0, 1.5, 1.0, 2.0, 2.0, 1.0, 1.0
_MD_NOISE = re.compile(r"!\[[^\]]*\]\([^)]*\)|\[!\[.*?\]\(.*?\)\]\(.*?\)|https?://\S+|<[^>]+>")
_CODE = re.compile(r"```.*?```", re.S)
STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the this to was were will with
you your we our can not use used using pharo smalltalk project repository github code load loading install metacello
baseline new spec image version see also more this how what which all any into license mit http https www com org""".split())


def clean_readme(text: str | None) -> str:
    if not text:
        return ""
    t = _CODE.sub(" ", text)
    t = _MD_NOISE.sub(" ", t)
    return re.sub(r"[ \t]+", " ", t)


def _excerpt(text: str, m: re.Match, width: int = 140) -> str:
    a = max(0, m.start() - width // 2)
    return re.sub(r"\s+", " ", text[a:m.end() + width // 2]).strip()


class DomainClassifier:
    def __init__(self, spec: dict, min_score: float = 2.0, framework_min_share: float = 0.02):
        self.version = str(spec.get("version", "0"))
        self.domains = spec["domains"]
        self.min_score = min_score
        self.fw_share = framework_min_share
        self.compiled = {
            d: {"keywords": [re.compile(k, re.I) for k in v.get("keywords") or []],
                "topics": {t.lower() for t in v.get("topics") or []},
                "deps": {x.lower() for x in v.get("dependencies") or []},
                "frameworks": set(v.get("frameworks") or []),
                "packages": [re.compile(p) for p in v.get("package_patterns") or []]}
            for d, v in self.domains.items()}

    def classify(self, repo: dict) -> dict:
        """``repo`` keys: full_name, name, description, readme, topics, dependencies,
        packages, framework_shares ({framework: share of global references})."""
        desc = repo.get("description") or ""
        readme = clean_readme(repo.get("readme"))
        topics = [t.lower() for t in repo.get("topics") or []]
        rich = bool(desc.strip() or readme.strip() or topics)
        labels = []
        for d, c in self.compiled.items():
            ev, score, sources = [], 0.0, set()
            for t in topics:
                if t in c["topics"]:
                    ev.append({"source": "topic", "file": None, "excerpt": t, "weight": W_TOPIC})
                    score += W_TOPIC
                    sources.add("topic")
            for rx in c["keywords"]:
                m = rx.search(desc)
                if m:
                    ev.append({"source": "description", "file": None, "excerpt": _excerpt(desc, m), "weight": W_DESC})
                    score += W_DESC
                    sources.add("description")
                m = rx.search(readme)
                if m:
                    ev.append({"source": "readme", "file": repo.get("readme_path") or "README",
                               "excerpt": _excerpt(readme, m), "weight": W_README})
                    score += W_README
                    sources.add("readme")
            for dep in repo.get("dependencies") or []:
                if dep.lower() in c["deps"]:
                    ev.append({"source": "baseline_dependency", "file": repo.get("baseline_file"), "excerpt": dep, "weight": W_DEP})
                    score += W_DEP
                    sources.add("dependency")
            for fw, share in (repo.get("framework_shares") or {}).items():
                if fw in c["frameworks"] and share >= self.fw_share:
                    ev.append({"source": "framework_references", "file": None,
                               "excerpt": f"{fw}: {share:.1%} of global references", "weight": W_FRAMEWORK})
                    score += W_FRAMEWORK
                    sources.add("framework")
            pk_hits = [p for p in repo.get("packages") or [] if any(rx.search(p) for rx in c["packages"])]
            if pk_hits:
                ev.append({"source": "package_names", "file": None, "excerpt": ", ".join(sorted(pk_hits)[:5]), "weight": W_PACKAGE})
                score += W_PACKAGE
                sources.add("packages")
            if not rich:
                name = repo.get("name") or ""
                for rx in c["keywords"]:
                    m = rx.search(name.replace("-", " "))
                    if m:
                        ev.append({"source": "repository_name_only", "file": None, "excerpt": name, "weight": W_NAME})
                        score += W_NAME
                        sources.add("name")
                        break
            # require at least two evidence items, or one strong (topic/dependency/framework) source
            strong_source = bool(sources & {"topic", "dependency", "framework"})
            if score >= self.min_score and (len(ev) >= 2 or strong_source) and sources != {"name"}:
                labels.append({"label": d, "score": round(score, 2), "score_kind": "uncalibrated_weight_sum",
                               "evidence": ev, "independent_sources": sorted(sources)})
        labels.sort(key=lambda x: -x["score"])
        if not labels:
            labels = [{"label": "unknown", "score": 0.0, "score_kind": "uncalibrated_weight_sum", "evidence": [],
                       "independent_sources": [], "note": "no domain rule reached the evidence threshold"}]
        return {"repository": repo.get("full_name"), "domain_labels": labels, "evidence_was_rich": rich}


def tokenize_terms(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", text.lower()) if w not in STOPWORDS]


def emergent_terms(unknown_docs: dict[str, str], all_docs: dict[str, str], top: int = 40) -> list[dict]:
    """Distinctive terms of repositories with no domain (TF-IDF vs the whole corpus),
    to help researchers propose new domain categories."""
    df = Counter()
    for doc in all_docs.values():
        df.update(set(tokenize_terms(doc)))
    n = max(1, len(all_docs))
    tf = Counter()
    repos_with = Counter()
    for doc in unknown_docs.values():
        toks = tokenize_terms(doc)
        tf.update(toks)
        repos_with.update(set(toks))
    scored = []
    for term, f in tf.items():
        if repos_with[term] < 2:
            continue
        idf = math.log(n / (1 + df[term])) + 1
        scored.append({"term": term, "tfidf": round(f * idf, 2), "unknown_repositories": repos_with[term],
                       "corpus_repositories": df[term]})
    scored.sort(key=lambda x: -x["tfidf"])
    return scored[:top]
