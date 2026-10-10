#!/usr/bin/env bash
# Finds a free GPU node anywhere on Grid'5000 and starts job.oar.sh on it. You do not have to
# look at which site or cluster is free.
#
#   ./submit-gpu.sh              run it on any Grid'5000 frontend, from this folder
#
# How it decides:
#   1. It asks the Grid'5000 API for the GPU clusters of every site and orders them: most GPU
#      memory per node first. Nothing is hard-coded: new clusters are found by themselves.
#   2. For each cluster it submits the job and waits a few seconds. If the job starts, it is
#      done. If the nodes are busy, the job is withdrawn and the next cluster is tried.
#   3. If everything is busy, the job is left in the queue of the site with the most GPU nodes
#      and starts there on the first one that becomes free.
# Each site has its own home directory, so before trying a cluster of another site the code
# and .env are copied there (same folder; results/ is not copied). The results are written on
# the site where the job runs: the script says which one.
# What the job itself does (models, number of repositories, ...) is set in job.oar.sh.

# ============================== SETTINGS ===================================================
# Sites to look at, separated by spaces. Empty = every site of Grid'5000.
# To continue a job that already ran on one site (its results are there): SITES="rennes"
SITES=""

# GPU memory a node needs, all its GPUs together, in GB. The three models of job.oar.sh need
# about 80 GB together (the biggest one, llama3.3:70b, needs 43 GB alone).
MIN_GPU_GB=90

# How long the node is reserved, in hours. A cluster that allows less gets its own maximum.
# The job continues where it stopped when it is submitted again on the same site.
WALLTIME_HOURS=72

# yes = also use the GPU clusters of the "default" queue (most sites outside Rennes).
# The Grid'5000 usage policy limits long jobs there during working hours: check that your
# reservation is allowed (https://www.grid5000.fr/w/Grid5000:UsagePolicy).
# no  = only the clusters of the "abaca" queue, which allows up to 7 days.
USE_DEFAULT_QUEUE="yes"

# Seconds to wait for the job to start on a cluster before trying the next one.
START_TIMEOUT=30

# What to do when every cluster is busy:
#   yes = leave the job in a queue: it starts on the first node that becomes free
#   no  = submit nothing and stop
WAIT_IF_BUSY="yes"

# yes = only list the clusters that would be tried, in order
DRY_RUN="no"
# ===========================================================================================

set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

command -v oarsub >/dev/null || {
  echo "oarsub was not found: run this script on a Grid'5000 frontend (for example: ssh rennes), from the repository folder."
  exit 1
}
case "$ROOT" in
  "$HOME"/*) REL="${ROOT#"$HOME"/}" ;;
  *) echo "The repository must be inside your home directory ($HOME) to be copied to the other sites."; exit 1 ;;
esac
[ -f "$ROOT/job.oar.sh" ] || { echo "job.oar.sh was not found in $ROOT."; exit 1; }
if [ ! -f "$ROOT/.env" ] || ! grep -q 'HF_TOKEN' "$ROOT/.env"; then
  echo "Note: no HF_TOKEN in .env: the job will run, but its results will not be uploaded to Hugging Face."
fi

# The site this frontend belongs to (frennes.rennes.grid5000.fr -> rennes)
LOCAL_SITE=$(hostname -f 2>/dev/null | awk -F. '$(NF-1) == "grid5000" { print $(NF-2) }')

# on SITE COMMAND: runs a shell command on the frontend of SITE
on() {
  local site="$1"; shift
  if [ "$site" = "$LOCAL_SITE" ]; then bash -c "$1"
  else ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$site" "$1"; fi
}

# prepare SITE: copies the code and .env to the same folder on SITE (once per run)
PREPARED=" "
prepare() {
  local site="$1"
  case "$PREPARED" in *" $site "*) return 0 ;; esac
  if [ "$site" != "$LOCAL_SITE" ]; then
    on "$site" "mkdir -p \"\$HOME/$REL\"" 2>/dev/null || return 1
    rsync -a --exclude results --exclude .venv --exclude oar-runs --exclude '__pycache__' \
          --exclude 'pharo-labels.*.out' --exclude 'pharo-labels.*.err' \
          -e "ssh -o BatchMode=yes -o ConnectTimeout=15" "$ROOT/" "$site:$REL/" 2>/dev/null || return 1
  fi
  PREPARED="$PREPARED$site "
}

# submit SITE QUEUE PROPERTY HOURS: submits job.oar.sh on one node, prints the job id.
# The script is passed as a command (not with -S), so the '#OAR' lines at the top of
# job.oar.sh, which name a CPU cluster, are not used here.
submit() {
  local out id d="\$HOME/$REL"
  out=$(on "$1" "cd \"$d\" && oarsub -q $2 -p \"$3\" -l host=1,walltime=$4:00:00 -n pharo-labels -O \"$d/pharo-labels.%jobid%.out\" -E \"$d/pharo-labels.%jobid%.err\" \"bash $d/job.oar.sh\"" 2>&1) \
    || { echo "$out" >&2; return 1; }
  id=$(printf '%s\n' "$out" | sed -n 's/^OAR_JOB_ID=\([0-9][0-9]*\).*/\1/p' | head -n 1)
  [ -n "$id" ] || { echo "$out" >&2; return 1; }
  echo "$id"
}

# state SITE ID: Waiting, Running, Error, ... ("" when it cannot be read)
state() {
  on "$1" "oarstat -s -j $2" 2>/dev/null | sed -n 's/^[0-9][0-9]*: *//p' | head -n 1
}

hours_for() {  # the cluster's own maximum when it is lower than WALLTIME_HOURS (0 = no maximum known)
  if [ "$1" -gt 0 ] && [ "$1" -lt "$WALLTIME_HOURS" ]; then echo "$1"; else echo "$WALLTIME_HOURS"; fi
}

where() {  # how to reach a site from here
  if [ "$1" = "$LOCAL_SITE" ]; then echo "here"; else echo "ssh $1"; fi
}

report() {  # report SITE ID QUEUE: where the job and its results are
  local prefix=""
  [ "$1" = "$LOCAL_SITE" ] || prefix="ssh $1 "
  echo "  follow it:   ${prefix}tail -f $REL/pharo-labels.$2.out"
  echo "  its state:   ${prefix}oarstat -j $2"
  echo "  stop it:     ${prefix}oardel $2"
  echo "Results: on the site $1 ($(where "$1")), in ~/$REL/results/"
  echo "To continue this run later (a job that hit its walltime), write SITES=\"$1\" at the top of this file and run it again."
  if [ "$3" = "default" ]; then
    echo "This cluster is in the default queue: check that a job of this length is allowed by the usage policy."
  fi
  echo "$1 $2" > "$ROOT/.gpu-job"
}

# ---- the GPU clusters, best first ----------------------------------------------------------
list=$(python3 "$ROOT/scripts/g5k_gpu_clusters.py" --min-gb "$MIN_GPU_GB" ${SITES:+--sites "$SITES"} ${G5K_API:+--api "$G5K_API"}) \
  || { echo "The list of GPU clusters could not be read from the Grid'5000 API (see above)."; exit 1; }
if [ "$USE_DEFAULT_QUEUE" != "yes" ]; then
  list=$(printf '%s\n' "$list" | awk -F'\t' '$3 != "default"')
fi
[ -n "$list" ] || { echo "No GPU cluster with $MIN_GPU_GB GB of GPU memory was found${SITES:+ on: $SITES}."; exit 1; }

if [ "$DRY_RUN" = "yes" ]; then
  echo "Clusters that would be tried, in this order:"
  printf '%s\n' "$list" | awk -F'\t' '{ printf "  %-11s %-12s %-8s %s node(s)  %s\n", $1, $2, $3, $5, $7 }'
  exit 0
fi

echo "Looking for a free GPU node on Grid'5000 (at least $MIN_GPU_GB GB of GPU memory, $WALLTIME_HOURS hours) …"
while IFS=$'\t' read -r site cluster queue property nodes max_hours description <&3; do
  printf '  %-11s %-12s %-42s ' "$site" "$cluster" "$description"
  if ! prepare "$site"; then
    echo "site not reachable"
    continue
  fi
  hours=$(hours_for "$max_hours")
  if ! id=$(submit "$site" "$queue" "$property" "$hours" 2>/dev/null); then
    echo "cannot be reserved (no access to it)"
    continue
  fi
  s=""
  for _ in $(seq 1 "$START_TIMEOUT"); do
    s=$(state "$site" "$id")
    case "$s" in
      Running|Launching|toLaunch)
        echo "free"
        echo
        echo "The job is running on $cluster, site $site (job $id, $hours hours)."
        report "$site" "$id" "$queue"
        exit 0 ;;
      Error|toError|Terminated|Finishing) break ;;
    esac
    sleep 1
  done
  on "$site" "oardel $id" >/dev/null 2>&1 || true
  case "$s" in
    Error|toError) echo "refused the job" ;;
    Terminated|Finishing)
      echo "the job stopped at once"
      echo "Read the error on $site: ~/$REL/pharo-labels.$id.err"
      exit 1 ;;
    *) echo "busy" ;;
  esac
done 3<<< "$list"

echo
if [ "$WAIT_IF_BUSY" != "yes" ]; then
  echo "Every GPU cluster is busy. Nothing was submitted (WAIT_IF_BUSY=no)."
  exit 1
fi
# Queue on the site with the most suitable nodes: the best chance that one becomes free soon.
# Only clusters of one queue can share a job; the queue with the most nodes on that site is used.
best=$(printf '%s\n' "$list" | awk -F'\t' '{ n[$1 "\t" $3] += $5 } END { for (k in n) print n[k] "\t" k }' | sort -k1,1nr -k2,2 | head -n 1)
site=$(printf '%s' "$best" | cut -f2); queue=$(printf '%s' "$best" | cut -f3)
clusters=$(printf '%s\n' "$list" | awk -F'\t' -v s="$site" -v q="$queue" -v a="'" '$1 == s && $3 == q && $4 ~ /^cluster=/ { printf "%s%s%s%s", sep, a, $2, a; sep = "," }')
hours=$(printf '%s\n' "$list" | awk -F'\t' -v s="$site" -v q="$queue" -v w="$WALLTIME_HOURS" '$1 == s && $3 == q && $6 > 0 && $6 < w { w = $6 } END { print w }')
if [ -z "$clusters" ] || ! prepare "$site" || ! id=$(submit "$site" "$queue" "cluster in ($clusters)" "$hours"); then
  echo "Every GPU cluster is busy and the job could not be queued on $site (see above). Try again later."
  exit 1
fi
echo "Every GPU cluster is busy. The job waits in the queue of the site $site (job $id):"
echo "it starts by itself on the first of these clusters that has a free node: $clusters"
echo "  planned start:  $([ "$site" = "$LOCAL_SITE" ] || echo "ssh $site ")oarstat -fj $id | grep -i scheduled"
report "$site" "$id" "$queue"
