#!/usr/bin/env python3
"""Lists the GPU clusters of every Grid'5000 site, best first, from the Grid'5000 Reference API.

Used by submit-gpu.sh. One line per cluster, tab-separated:
    site  cluster  queue  oar_property  nodes  max_walltime_hours  description

A cluster is listed when at least one of its nodes is x86_64 and has Nvidia GPUs with
--min-gb of memory in total. When only some nodes qualify, oar_property names those nodes.
Order: total GPU memory of a node, then memory per GPU; cards with an old driver environment
(Tesla P100/V100) come after the others.

The API answers without a password from inside Grid'5000 (frontends and nodes).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import urllib.request

API = "https://api.grid5000.fr/stable"
OLD_CARDS = ("P100", "V100", "K40", "K80", "M40", "GTX", "1080")


def get(base: str, path: str):
    req = urllib.request.Request(base + path, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _items(v) -> list:
    """The API gives lists ({"items": [...]}) or, in deep answers, dictionaries keyed by uid."""
    if isinstance(v, dict):
        if isinstance(v.get("items"), list):
            return v["items"]
        return [dict(x, uid=x.get("uid", k)) for k, x in v.items() if isinstance(x, dict)]
    return v if isinstance(v, list) else []


def site_clusters(base: str, site: str) -> list[tuple[dict, list[dict]]]:
    try:  # one request for the whole site when the API supports it
        deep = _items(get(base, f"/sites/{site}?deep=true").get("clusters"))
        if deep and all("nodes" in c for c in deep):
            return [(c, _items(c["nodes"])) for c in deep]
    except Exception:  # noqa: BLE001 - fall back to one request per cluster
        pass
    out = []
    for c in _items(get(base, f"/sites/{site}/clusters")):
        out.append((c, _items(get(base, f"/sites/{site}/clusters/{c['uid']}/nodes"))))
    return out


def node_gpus(node: dict) -> tuple[float, float, int, str]:
    """(total GiB, GiB of the smallest card, number of cards, model) of the Nvidia GPUs of a node."""
    if (node.get("architecture") or {}).get("platform_type") != "x86_64":
        return 0.0, 0.0, 0, ""
    cards = [g for g in _items(node.get("gpu_devices")) if "nvidia" in str(g.get("vendor", "")).lower()
             and g.get("memory")]
    if not cards:
        return 0.0, 0.0, 0, ""
    mem = [float(g["memory"]) / 2 ** 30 for g in cards]
    return sum(mem), min(mem), len(cards), str(cards[0].get("model", "GPU"))


def site_rows(base: str, site: str, min_gb: float) -> list[dict]:
    rows = []
    for cluster, nodes in site_clusters(base, site):
        good = [(n, node_gpus(n)) for n in nodes]
        good = [(n, g) for n, g in good if g[0] >= min_gb]
        if not good:
            continue
        name = cluster["uid"]
        total, per_card, count, model = min((g for _, g in good), key=lambda g: g[0])
        jobs = (good[0][0].get("supported_job_types") or {})
        queues = jobs.get("queues") or cluster.get("queues") or []
        queue = "default" if "default" in queues else ("abaca" if {"abaca", "production"} & set(queues) else "default")
        hosts = sorted(f"{n['uid']}.{site}.grid5000.fr" for n, _ in good)
        prop = f"cluster='{name}'" if len(good) == len(nodes) else "host in (%s)" % ",".join(f"'{h}'" for h in hosts)
        rows.append({"site": site, "cluster": name, "queue": queue, "property": prop, "nodes": len(good),
                     "max_walltime_hours": int((jobs.get("max_walltime") or 0) // 3600),
                     "total": total, "per_card": per_card, "old": any(k in model for k in OLD_CARDS),
                     "description": f"{count} x {model} ({per_card:.0f} GB each, {total:.0f} GB)"})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-gb", type=float, default=90, help="GPU memory a node needs in total (default 90)")
    ap.add_argument("--sites", default="", help="Only these sites, separated by spaces (default: all)")
    ap.add_argument("--api", default=API, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        sites = args.sites.split() or sorted(s["uid"] for s in _items(get(args.api, "/sites")))
    except Exception as e:  # noqa: BLE001
        print(f"Cannot reach the Grid'5000 API ({args.api}): {e}", file=sys.stderr)
        return 2
    rows = []

    def one(site):
        try:
            return site_rows(args.api, site, args.min_gb)
        except Exception as e:  # noqa: BLE001 - a site in maintenance must not hide the others
            print(f"site {site} skipped: {e}", file=sys.stderr)
            return []

    with ThreadPoolExecutor(max_workers=8) as pool:
        for r in pool.map(one, sites):
            rows += r
    rows.sort(key=lambda r: (r["old"], -r["total"], -r["per_card"], r["site"], r["cluster"]))
    for r in rows:
        print("\t".join(str(r[k]) for k in ("site", "cluster", "queue", "property", "nodes", "max_walltime_hours", "description")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
