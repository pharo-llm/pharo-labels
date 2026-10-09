#!/usr/bin/env bash
# Kept for compatibility: the workflow is now split into three steps.
#   ./1-mining.sh        GitHub mining, parsing, rule-based labels, annotation sample
#   ./2-ai-labelling.sh  label the sample with language models (models chosen at the top of that file)
#   ./3-results.sh       taxonomy, statistics, benchmark coverage, reports
exec "$(dirname "$0")/1-mining.sh" "$@"
