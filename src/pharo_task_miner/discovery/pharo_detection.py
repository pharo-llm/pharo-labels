"""Module 2 — distinguish Pharo repositories from other Smalltalk dialects.

Inputs are a file listing (GitHub tree API or a local checkout), the README,
CI/project configuration files and repository metadata. Every decision keeps
its supporting evidence *and* its counter-evidence. Possible statuses:
``confirmed_pharo``, ``probable_pharo``, ``uncertain`` (eligible for manual
review, never silently discarded) and ``non_pharo``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path, PurePosixPath
import re

PHARO_WORD = re.compile(r"\bpharo\b", re.I)
PHARO_VERSION = re.compile(r"\bpharo[\s\-_]?(\d{1,2}(?:\.\d+)?)\b", re.I)
OTHER_DIALECTS = {
    "gemstone": re.compile(r"\bgem\s?stone(?:/s)?\b", re.I),
    "squeak": re.compile(r"\bsqueak\b", re.I),
    "cuis": re.compile(r"\bcuis\b", re.I),
    "visualworks": re.compile(r"\bvisual\s?works\b|\bcincom\b", re.I),
    "va_smalltalk": re.compile(r"\bva smalltalk\b|\bvast\b|\bvisualage\b", re.I),
    "dolphin": re.compile(r"\bdolphin smalltalk\b", re.I),
    "gnu_smalltalk": re.compile(r"\bgnu smalltalk\b|\bgst\b", re.I),
    "glamorous_toolkit": re.compile(r"\bglamorous toolkit\b|\bgtoolkit\b", re.I),
}
CI_PHARO = re.compile(r"pharo(?:64|32)?[-_ ]?(?:alpha|stable|\d+(?:\.\d+)?)|smalltalk-image:\s*['\"]?pharo|#smalltalk\s*:\s*'Pharo", re.I)
CI_OTHER = re.compile(r"(squeak|gemstone|cuis)[-_ ]?\d|smalltalk-image:\s*['\"]?(squeak|gemstone)", re.I)
CI_FILES = (".smalltalk.ston", ".travis.yml", ".gitlab-ci.yml", ".project", "appveyor.yml")
SOURCE_SUFFIXES = (".st", ".cs")


@dataclass
class Listing:
    paths: list[str]
    sizes: dict[str, int] = field(default_factory=dict)
    truncated: bool = False
    tree_sha: str | None = None
    source: str = "unknown"


@dataclass
class Verification:
    status: str
    evidence: list[dict]
    counter_evidence: list[dict]
    summary: dict
    pharo_versions: list[str]
    source: str

    def to_dict(self):
        return {"pharo_status": self.status, "evidence": self.evidence, "counter_evidence": self.counter_evidence,
                "summary": self.summary, "pharo_versions": self.pharo_versions, "verification_source": self.source}


def summarize_listing(listing: Listing) -> dict:
    paths = listing.paths
    st = [p for p in paths if p.endswith(SOURCE_SUFFIXES)]
    tonel = [p for p in paths if p.endswith((".class.st", ".extension.st", ".trait.st"))]
    filetree = [p for p in st if "/instance/" in p or "/class/" in p]
    packages = {str(PurePosixPath(p).parent) for p in paths if p.endswith("/package.st") or p == "package.st"}
    packages |= {seg for p in paths for seg in [next((s for s in p.split("/") if s.endswith(".package")), None)] if seg}
    tests = [p for p in st if re.search(r"Tests?\b|Test\.class|Tests?[-.]", p)]
    return {
        "n_files": len(paths), "n_source_files": len(st), "n_tonel_files": len(tonel),
        "n_filetree_method_files": len(filetree), "n_packages": len(packages),
        "n_test_source_files": len(tests), "source_bytes": sum(listing.sizes.get(p, 0) for p in st),
        "has_tests": bool(tests), "tree_truncated": listing.truncated, "tree_sha": listing.tree_sha,
        "has_readme": any(PurePosixPath(p).name.lower().startswith("readme") for p in paths),
        "has_license_file": any(PurePosixPath(p).name.lower().startswith(("license", "licence", "copying")) for p in paths),
    }


def _excerpt(text: str, m: re.Match, width: int = 90) -> str:
    a = max(0, m.start() - width // 2)
    return re.sub(r"\s+", " ", text[a:m.end() + width // 2]).strip()


def verify(listing: Listing | None, readme: str | None, ci_files: dict[str, str], topics: list[str],
           description: str | None, language: str | None) -> Verification:
    ev, counter = [], []

    def add(kind, strength, detail, target=ev):
        target.append({"kind": kind, "strength": strength, "detail": detail})

    summary = summarize_listing(listing) if listing else {}
    paths = listing.paths if listing else []
    # -- file-structure evidence
    baselines = [p for p in paths if re.search(r"(^|/)BaselineOf[^/]*\.class\.st$", p) or re.search(r"BaselineOf[^/]*\.package/", p)]
    if baselines:
        add("baseline", "medium", baselines[0])
    if any(p.endswith(".project") and p.count("/") == 0 for p in paths):
        add("iceberg_project_file", "strong", ".project (Iceberg project metadata)")
    if summary.get("n_tonel_files"):
        add("tonel_sources", "medium", f"{summary['n_tonel_files']} Tonel files")
    if summary.get("n_filetree_method_files"):
        add("filetree_sources", "weak", f"{summary['n_filetree_method_files']} FileTree method files")
    if any(p.endswith(".smalltalk.ston") for p in paths):
        add("smalltalkci_config", "weak", ".smalltalk.ston")
    for p in paths:
        if p.endswith(".pck.st"):
            add("cuis_package_file", "strong", p, counter)
            break
    if any(p.endswith((".gs", ".tpz")) for p in paths) or any(p.startswith("rowan/") for p in paths):
        add("gemstone_files", "medium", "GemStone topaz/Rowan files", counter)
    if any(p.endswith((".pcl", ".pst")) for p in paths):
        add("visualworks_parcels", "strong", "VisualWorks parcel files", counter)
    if any(p.endswith((".pac", ".cls")) for p in paths):
        add("dolphin_packages", "medium", "Dolphin package files", counter)
    # -- CI / project configuration
    for path, text in (ci_files or {}).items():
        if not text:
            continue
        m = CI_PHARO.search(text)
        if m:
            add("ci_pharo_image", "strong", f"{path}: {_excerpt(text, m, 40)}")
        m2 = CI_OTHER.search(text)
        if m2:
            add("ci_other_dialect", "weak", f"{path}: {_excerpt(text, m2, 40)}", counter)
        if path.endswith(".project") and "srcDirectory" in text:
            add("iceberg_project_file", "strong", f"{path} declares srcDirectory")
    # -- documentation
    text = readme or ""
    versions = sorted({v for v in PHARO_VERSION.findall(text + " " + (description or ""))},
                      key=lambda v: [int(x) for x in v.split(".")])
    m = PHARO_WORD.search(text)
    if m:
        add("readme_mentions_pharo", "medium", _excerpt(text, m))
    if versions:
        add("pharo_versions_mentioned", "medium", ", ".join(versions))
    if re.search(r"Metacello\s+new", text):
        add("metacello_install_snippet", "weak", "README contains a Metacello load script")
    if description and PHARO_WORD.search(description):
        add("description_mentions_pharo", "medium", description[:200])
    if any(t.lower() == "pharo" or t.lower().startswith("pharo-") for t in topics or []):
        add("topic_pharo", "medium", "topic:pharo")
    for dialect, rx in OTHER_DIALECTS.items():
        dm = rx.search(text + " " + (description or ""))
        if dm and not m:
            add(f"mentions_{dialect}", "medium", _excerpt(text + " " + (description or ""), dm), counter)
        elif dm:
            add(f"also_mentions_{dialect}", "weak", _excerpt(text + " " + (description or ""), dm), counter)
    # -- decision
    strong = sum(e["strength"] == "strong" for e in ev)
    medium = sum(e["strength"] == "medium" for e in ev)
    weak = sum(e["strength"] == "weak" for e in ev)
    c_strong = sum(e["strength"] == "strong" for e in counter)
    c_medium = sum(e["strength"] == "medium" for e in counter)
    has_source = bool(summary.get("n_source_files")) if listing else None
    if listing is None:
        status = "uncertain"
        counter.append({"kind": "no_file_listing", "strength": "info",
                        "detail": "No file listing available; decision needs the tree API or a clone"})
    elif not has_source:
        status = "non_pharo"
        counter.append({"kind": "no_smalltalk_sources", "strength": "strong", "detail": "No .st source files in listing"})
    elif c_strong and not strong:
        status = "non_pharo" if not medium else "uncertain"
    elif strong >= 1 or medium >= 2:
        status = "confirmed_pharo" if not (c_strong or c_medium) or strong >= 1 else "uncertain"
    elif medium == 1 or (weak >= 1 and (language or "").lower() == "smalltalk"):
        status = "probable_pharo" if not (c_strong or c_medium) else "uncertain"
    else:
        status = "uncertain"
    if summary.get("tree_truncated"):
        counter.append({"kind": "tree_truncated", "strength": "info", "detail": "GitHub returned a truncated tree"})
    return Verification(status, ev, counter, summary, versions, listing.source if listing else "metadata_only")


def local_listing(root: Path) -> Listing:
    paths, sizes = [], {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for fn in filenames:
            full = Path(dirpath) / fn
            rel = full.relative_to(root).as_posix()
            paths.append(rel)
            try:
                sizes[rel] = full.stat().st_size
            except OSError:
                pass
    return Listing(sorted(paths), sizes, False, None, "local_clone")


def local_readme_and_ci(root: Path, max_ci: int = 5) -> tuple[str | None, dict[str, str]]:
    readme = None
    for cand in sorted(root.iterdir()) if root.exists() else []:
        if cand.is_file() and cand.name.lower().startswith("readme"):
            readme = cand.read_text(encoding="utf-8", errors="replace")
            break
    ci = {}
    for name in CI_FILES:
        p = root / name
        if p.is_file():
            ci[name] = p.read_text(encoding="utf-8", errors="replace")[:20000]
    wf = root / ".github" / "workflows"
    if wf.is_dir():
        for p in sorted(wf.iterdir())[:max_ci]:
            if p.suffix in (".yml", ".yaml"):
                ci[f".github/workflows/{p.name}"] = p.read_text(encoding="utf-8", errors="replace")[:20000]
    return readme, ci


def remote_listing(client, full_name: str, branch: str | None) -> Listing | None:
    """Recursive file listing of the default branch in a single API request."""
    t = client.get(f"/repos/{full_name}/git/trees/{branch or 'HEAD'}", {"recursive": "1"})
    if not isinstance(t.data, dict):
        return None  # empty, deleted or blocked repository
    paths, sizes = [], {}
    for e in t.data.get("tree", []):
        if e.get("type") == "blob":
            paths.append(e["path"])
            sizes[e["path"]] = e.get("size", 0)
    return Listing(sorted(paths), sizes, bool(t.data.get("truncated")), t.data.get("sha"), "github_tree_api")


RAW = "application/vnd.github.raw+json"


def remote_readme(client, full_name: str, listing: Listing | None) -> tuple[str | None, str | None]:
    readme_path = next((p for p in listing.paths if "/" not in p and p.lower().startswith("readme")), None) if listing else None
    if listing is not None and readme_path is None:
        return None, None  # no README: skip the request
    r = client.get(f"/repos/{full_name}/readme", accept=RAW)
    return (r.data if isinstance(r.data, str) else None), readme_path


def remote_ci(client, full_name: str, listing: Listing, max_ci: int = 5) -> dict[str, str]:
    wanted = [p for p in listing.paths if p in CI_FILES]
    wanted += [p for p in listing.paths if p.startswith(".github/workflows/") and p.endswith((".yml", ".yaml"))][:max_ci]
    ci = {}
    for p in wanted:
        resp = client.get(f"/repos/{full_name}/contents/{p}", accept=RAW)
        if isinstance(resp.data, str):
            ci[p] = resp.data[:20000]
    return ci
