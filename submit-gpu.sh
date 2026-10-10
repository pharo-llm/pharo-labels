#!/usr/bin/env bash
# Grid'5000 GPU checker: shows, site by site, which GPU nodes are free right now, and offers to
# submit a job script on one of them.
#
# This file is self-contained. Copy it into any project: it needs nothing else from this one.
# Run it on a Grid'5000 frontend (any site), from the folder of your project.
#
#   ./submit-gpu.sh                    show the free GPUs of every site, then ask where to submit
#   ./submit-gpu.sh --list             only show them
#   ./submit-gpu.sh --yes              submit on the strongest free one without asking
#   ./submit-gpu.sh --cluster chuc     submit on this cluster (the job waits if it is busy)
#   ./submit-gpu.sh my-job.sh          submit this script instead of the one set below
#
# What you see comes from the Grid'5000 API, at the moment you run it: every site, every GPU
# cluster, how many of its nodes nobody is using. No cluster name is written in this file.
# A job reserves one whole node, with all its GPUs.
#
# Each site has its own home directory. When you choose a cluster of another site, the folder
# of the job script is copied there first (same path), and the job runs and writes there.

# ============================== SETTINGS ===================================================
# The job script to submit, relative to this file (or give another one on the command line).
JOB_SCRIPT="job.oar.sh"

# How long the node is reserved, in hours. A cluster that allows less gets its own maximum.
WALLTIME_HOURS=72

# Sites to look at, separated by spaces. Empty = every site of Grid'5000.
SITES=""

# Only show nodes whose GPUs have at least this much memory together, in GB. 0 = show all.
MIN_GPU_GB=0

# Folders and files that are not copied to another site (big or rebuilt there).
DO_NOT_COPY="results .venv oar-runs __pycache__ node_modules *.out *.err"

# Seconds to wait to see whether the job started.
START_TIMEOUT=30
# ===========================================================================================

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

MODE="ask"; WANTED=""
while [ $# -gt 0 ]; do
  case "$1" in
    --list) MODE="list" ;;
    --yes|-y) MODE="yes" ;;
    --cluster) WANTED="${2:-}"; shift ;;
    -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "Unknown option $1 (see --help)."; exit 1 ;;
    *) JOB_SCRIPT="$1" ;;
  esac
  shift
done

case "$JOB_SCRIPT" in /*) JOB="$JOB_SCRIPT" ;; *) JOB="$HERE/$JOB_SCRIPT" ;; esac
[ -f "$JOB" ] || [ "$MODE" = "list" ] || { echo "The job script $JOB was not found (set JOB_SCRIPT at the top of this file, or give it as argument)."; exit 1; }
ROOT="$(cd "$(dirname "$JOB")" 2>/dev/null && pwd || echo "$HERE")"
JOB_NAME="$(basename "$ROOT")"

# ---- what Grid'5000 has, and what is free --------------------------------------------------
# One line per site (S) and per group of identical GPU nodes (C), tab-separated, strongest first.
discover() {
  python3 - "$MIN_GPU_GB" "${G5K_API:-https://api.grid5000.fr/stable}" $SITES <<'PY'
import json, os, ssl, subprocess, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

MIN_GB, API, SITES = float(sys.argv[1]), sys.argv[2], sys.argv[3:]
OLD_CARDS = ("P100", "V100", "K40", "K80", "M40", "GTX", "1080")


def ssl_context():
    # A conda or virtualenv Python has its own certificate list, which does not know the
    # Grid'5000 authority: add the certificates of the system.
    ctx = ssl.create_default_context()
    for bundle in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt"):
        if os.path.exists(bundle):
            try:
                ctx.load_verify_locations(cafile=bundle)
            except (OSError, ssl.SSLError):
                pass
    return ctx


def get(path):
    url = API + path
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30, context=ssl_context() if url.startswith("https") else None) as r:
            return json.load(r)
    except urllib.error.URLError as e:
        if not isinstance(getattr(e, "reason", None), ssl.SSLError):
            raise
        out = subprocess.run(["curl", "-sSf", "--max-time", "30", "-H", "Accept: application/json", url],
                             capture_output=True, text=True)  # the system's curl knows the certificates
        if out.returncode != 0:
            raise RuntimeError(f"{e.reason}; curl also failed: {out.stderr.strip()}") from e
        return json.loads(out.stdout)


def items(v):
    """The API gives lists ({"items": [...]}) or, in deep answers, dictionaries keyed by uid."""
    if isinstance(v, dict):
        if isinstance(v.get("items"), list):
            return v["items"]
        return [dict(x, uid=x.get("uid", k)) for k, x in v.items() if isinstance(x, dict)]
    return v if isinstance(v, list) else []


def clusters_of(site):
    try:  # one request for the whole site when the API supports it
        deep = items(get(f"/sites/{site}?deep=true").get("clusters"))
        if deep and all("nodes" in c for c in deep):
            return [(c, items(c["nodes"])) for c in deep]
    except Exception:
        pass
    return [(c, items(get(f"/sites/{site}/clusters/{c['uid']}/nodes"))) for c in items(get(f"/sites/{site}/clusters"))]


def gpus(node):
    """(number of cards, model, GB per card, GB in total) of the Nvidia GPUs of an x86 node."""
    if (node.get("architecture") or {}).get("platform_type", "x86_64") != "x86_64":
        return 0, "", 0, 0
    cards = [g for g in items(node.get("gpu_devices")) if not g.get("vendor") or "nvidia" in str(g["vendor"]).lower()]
    if not cards:
        return 0, "", 0, 0
    mem = [float(g.get("memory") or 0) / 2 ** 30 for g in cards]
    return len(cards), str(cards[0].get("model", "GPU")), round(min(mem)), round(sum(mem))


def free_nodes(site):
    """{node: hours until its next reservation (0 = none planned)} for the nodes nobody uses now."""
    now, out = time.time(), {}
    for n in items(get(f"/sites/{site}/status").get("nodes")):
        if str(n.get("hard", "alive")).lower() != "alive" or str(n.get("soft", "")).lower() != "free":
            continue
        upcoming = []
        for r in n.get("reservations") or []:
            start = r.get("started_at") or r.get("scheduled_at") or r.get("start_time") or 0
            if start and start <= now:
                upcoming = None  # a job is on it
                break
            if start:
                upcoming.append((start - now) / 3600)
        if upcoming is not None:
            out[str(n["uid"]).split(".")[0]] = min(upcoming) if upcoming else 0.0
    return out


def site_rows(site):
    try:
        clusters = clusters_of(site)
    except Exception as e:
        return [("S", site, "unreachable", str(e)[:80])], []
    try:
        free = free_nodes(site)
    except Exception:
        free = None  # the site does not say which nodes are free
    rows = []
    for cluster, nodes in clusters:
        groups = {}
        for n in nodes:  # a cluster may mix cards (2 x P100 on some nodes, 2 x V100 on others)
            g = gpus(n)
            if g[0] and g[3] >= MIN_GB:
                groups.setdefault(g, []).append(n)
        for (count, model, per_card, total), members in groups.items():
            first = members[0]
            jobs = first.get("supported_job_types") or {}
            queues = jobs.get("queues") or cluster.get("queues") or []
            # the queue under the name the site gives it (the production queue is "abaca" on some)
            queue = next((q for q in ("default", "abaca", "production") if q in queues), "default")
            prop = (f"cluster='{cluster['uid']}'" if len(members) == len(nodes)
                    else "host IN (%s)" % ", ".join(sorted(n["uid"] for n in members)))
            exotic = any(n.get("exotic") for n in members) or cluster.get("exotic")
            if free is None:
                n_free, text = "?", "state unknown"
            else:
                mine = [free[n["uid"]] for n in members if n["uid"] in free]
                n_free = len(mine)
                limit = max(mine) if mine and all(mine) else 0  # every free node has a reservation coming
                text = ("in use" if not mine else f"{n_free} of {len(members)} nodes free"
                        + (f", for {limit:.0f} h" if limit else ""))
            arch, ram = first.get("architecture") or {}, (first.get("main_memory") or {}).get("ram_size") or 0
            cpu = (first.get("processor") or {}).get("model") or ""
            machine = ", ".join(x for x in (f"{arch.get('nb_cores')} cores" if arch.get("nb_cores") else "", cpu,
                                            f"{ram / 2 ** 30:.0f} GB RAM" if ram else "") if x) or "-"
            size = f"{per_card} GB each, {total} GB in total" if total else "size not given"
            rows.append({"sort": (any(k in model for k in OLD_CARDS), -total, -per_card, site, cluster["uid"]),
                         "line": ("C", site, cluster["uid"], queue, prop, len(members),
                                  int((jobs.get("max_walltime") or 0) // 3600), "exotic" if exotic else "-",
                                  n_free, text, f"{count} x {model} ({size})", machine)})
    return [("S", site, "ok", "-")], rows


try:
    sites = SITES or sorted(s["uid"] for s in items(get("/sites")))
except Exception as e:
    print(f"The Grid'5000 API does not answer ({API}): {e}", file=sys.stderr)
    sys.exit(2)
heads, rows = [], []
with ThreadPoolExecutor(max_workers=8) as pool:
    for h, r in pool.map(site_rows, sites):
        heads += h
        rows += r
for line in heads + [r["line"] for r in sorted(rows, key=lambda r: r["sort"])]:
    print("\t".join(str(x) for x in line))
PY
}

echo "Asking Grid'5000 which GPUs are free …"
DATA=$(discover) || { echo "The GPU clusters could not be read from the Grid'5000 API (see above)."; exit 1; }

# The choices, strongest first: the groups of GPU nodes with a free node, then those of the
# sites that do not say which nodes are free
CHOICES=()
while IFS= read -r line; do [ -n "$line" ] && CHOICES+=("$line"); done \
  < <(printf '%s\n' "$DATA" | awk -F'\t' '$1 == "C" && $9 != "0" && $9 != "?"';
      printf '%s\n' "$DATA" | awk -F'\t' '$1 == "C" && $9 == "?"')
field() { printf '%s' "$1" | cut -f"$2"; }

echo
for site in $(printf '%s\n' "$DATA" | awk -F'\t' '$1 == "S" { print $2 }' | sort); do
  state=$(printf '%s\n' "$DATA" | awk -F'\t' -v s="$site" '$1 == "S" && $2 == s { print $3 }')
  all=$(printf '%s\n' "$DATA" | awk -F'\t' -v s="$site" '$1 == "C" && $2 == s' | wc -l | tr -d ' ')
  free=$(printf '%s\n' "$DATA" | awk -F'\t' -v s="$site" '$1 == "C" && $2 == s && $9 != "0" && $9 != "?"' | wc -l | tr -d ' ')
  unknown=$(printf '%s\n' "$DATA" | awk -F'\t' -v s="$site" '$1 == "C" && $2 == s && $9 == "?"' | wc -l | tr -d ' ')
  if [ "$state" != "ok" ]; then
    printf '%-12s the site does not answer\n' "$site"
  elif [ "$all" = "0" ]; then
    printf '%-12s no GPU\n' "$site"
  elif [ "$unknown" != "0" ]; then
    printf '%-12s %s GPU cluster(s); the site does not say which are free\n' "$site" "$all"
  elif [ "$free" = "0" ]; then
    printf '%-12s no GPU free  (%s GPU cluster(s), all in use)\n' "$site" "$all"
  else
    printf '%-12s %s GPU cluster(s) free  (of %s)\n' "$site" "$free" "$all"
  fi
  i=0
  for c in ${CHOICES[@]+"${CHOICES[@]}"}; do
    i=$((i + 1))
    [ "$(field "$c" 2)" = "$site" ] || continue
    hours=$(field "$c" 7); if [ "$hours" = "0" ]; then hours="no time limit given"; else hours="up to $hours h"; fi
    printf '   [%d] %-12s %s\n' "$i" "$(field "$c" 3)" "$(field "$c" 11)"
    printf '       %s  |  %s  |  queue %s, %s%s\n' "$(field "$c" 10)" "$(field "$c" 12)" "$(field "$c" 4)" "$hours" \
           "$([ "$(field "$c" 8)" = "-" ] || echo ", exotic")"
  done
done
echo
[ "$MODE" != "list" ] || exit 0

# ---- which one -----------------------------------------------------------------------------
pick=""
if [ -n "$WANTED" ]; then
  pick=$(printf '%s\n' "$DATA" | awk -F'\t' -v c="$WANTED" '$1 == "C" && $3 == c' | sort -t$'\t' -k9,9nr | head -n 1)
  [ -n "$pick" ] || { echo "No GPU cluster named $WANTED was found."; exit 1; }
elif [ "${#CHOICES[@]}" -eq 0 ]; then
  echo "No GPU node is free at the moment. Run this again later, or wait in the queue of a cluster:"
  echo "  $0 --cluster <name>"
  exit 1
elif [ "$MODE" = "yes" ]; then
  pick="${CHOICES[0]}"
else
  if ! { : < /dev/tty; } 2>/dev/null; then
    echo "No terminal to ask on: nothing was submitted. Use --yes to submit on the strongest free one."
    exit 1
  fi
  printf 'Submit %s (%s hours) on which one?  [1-%d, Enter = 1, n = nothing] ' "$(basename "$JOB")" "$WALLTIME_HOURS" "${#CHOICES[@]}"
  read -r answer < /dev/tty || answer="n"
  answer="${answer:-1}"
  case "$answer" in
    *[!0-9]*) echo "Nothing was submitted."; exit 0 ;;
  esac
  if [ "$answer" -lt 1 ] || [ "$answer" -gt "${#CHOICES[@]}" ]; then echo "There is no choice $answer. Nothing was submitted."; exit 1; fi
  pick="${CHOICES[$((answer - 1))]}"
fi
site=$(field "$pick" 2); cluster=$(field "$pick" 3); queue=$(field "$pick" 4); property=$(field "$pick" 5)
max_hours=$(field "$pick" 7); type=$(field "$pick" 8)
hours="$WALLTIME_HOURS"
if [ "$max_hours" -gt 0 ] && [ "$max_hours" -lt "$hours" ]; then hours="$max_hours"; fi

# ---- submit ---------------------------------------------------------------------------------
command -v oarsub >/dev/null || { echo "oarsub was not found: run this on a Grid'5000 frontend (for example: ssh rennes)."; exit 1; }
case "$ROOT" in
  "$HOME"/*) REL="${ROOT#"$HOME"/}" ;;
  *) echo "The job script must be inside your home directory ($HOME) to be copied to another site."; exit 1 ;;
esac
JOB_REL="$REL/$(basename "$JOB")"
LOCAL_SITE=$(hostname -f 2>/dev/null | awk -F. '$(NF-1) == "grid5000" { print $(NF-2) }')
on() {  # on COMMAND: runs a shell command on the frontend of the chosen site
  if [ "$site" = "$LOCAL_SITE" ]; then bash -c "$1"
  else ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$site" "$1"; fi
}
prefix=""; [ "$site" = "$LOCAL_SITE" ] || prefix="ssh $site "

if [ "$site" != "$LOCAL_SITE" ]; then
  echo "Copying $ROOT to the site $site …"
  excludes=()
  set -f; for x in $DO_NOT_COPY; do excludes+=(--exclude "$x"); done; set +f
  on "mkdir -p \"\$HOME/$REL\"" && rsync -a ${excludes[@]+"${excludes[@]}"} \
      -e "ssh -o BatchMode=yes -o ConnectTimeout=15" "$ROOT/" "$site:$REL/" \
    || { echo "The site $site could not be reached. Nothing was submitted."; exit 1; }
fi

# The script is passed as a command (not with -S): '#OAR' lines inside it are not used, the
# cluster, queue and walltime chosen here are.
d="\$HOME/$REL"; t=""; [ "$type" = "-" ] || t="-t $type "
out=$(on "cd \"$d\" && oarsub -q $queue ${t}-p \"$property\" -l host=1,walltime=$hours:00:00 -n \"$JOB_NAME\" -O \"$d/$JOB_NAME.%jobid%.out\" -E \"$d/$JOB_NAME.%jobid%.err\" \"bash \$HOME/$JOB_REL\"" 2>&1) \
  || { echo "$out"; echo "The job was refused by $cluster (see above). Nothing was submitted."; exit 1; }
id=$(printf '%s\n' "$out" | sed -n 's/^OAR_JOB_ID=\([0-9][0-9]*\).*/\1/p' | head -n 1)
[ -n "$id" ] || { echo "$out"; echo "The job id could not be read (see above)."; exit 1; }

echo "Submitted on $cluster, site $site: job $id, $hours hours. Waiting to see it start …"
s=""
for _ in $(seq 1 "$START_TIMEOUT"); do
  s=$(on "oarstat -s -j $id" 2>/dev/null | sed -n 's/^[0-9][0-9]*: *//p' | head -n 1)
  case "$s" in Running|Launching|toLaunch|Error|toError|Terminated|Finishing) break ;; esac
  sleep 1
done
echo
case "$s" in
  Running|Launching|toLaunch) echo "The job is running." ;;
  Error|toError|Terminated|Finishing) echo "The job stopped at once (state: $s). Read: ${prefix}cat $REL/$JOB_NAME.$id.err" ;;
  *) echo "The job has not started yet: it waits in the queue and starts by itself when a node is free."
     echo "  planned start:  ${prefix}oarstat -fj $id | grep -i scheduled" ;;
esac
echo "  follow it:   ${prefix}tail -f $REL/$JOB_NAME.$id.out"
echo "  its state:   ${prefix}oarstat -j $id"
echo "  stop it:     ${prefix}oardel $id"
echo "The job runs and writes on the site $site, in ~/$REL/"
if [ "$queue" = "default" ]; then
  echo "This cluster is in the default queue, where the usage policy limits long jobs. Check with: ${prefix}usagepolicycheck -t"
fi
