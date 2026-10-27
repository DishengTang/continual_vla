#!/usr/bin/env bash
# Run a job file (see experiments/README.md). Each line is
#     <name>|<prerequisite path, or ->|<arguments for python>
# A job starts once its prerequisite exists (an output of an earlier job), in file order, at most MAX
# at a time; $CVLA_RUNS and other variables in a line are expanded when it starts.
#
#     source scripts/env.sh && MAX=2 bash experiments/run_jobs.sh experiments/01_main_study.jobs
#
# Started markers and one log per job are kept in $CVLA_RUNS/jobs/<job file>/, so running the same
# file again skips jobs that already started. Dependents of a failed job wait: check the logs.
set -u
JOBS=$(realpath "$1")
MAX=${MAX:-2}
STATE=$CVLA_RUNS/jobs/$(basename "$JOBS" .jobs)
mkdir -p "$STATE/started" "$STATE/pids" "$STATE/logs"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

while true; do
    running=0
    for f in "$STATE"/pids/*; do
        [ -e "$f" ] || continue
        if kill -0 "$(cat "$f")" 2>/dev/null; then
            running=$((running + 1))
        else
            echo "$(date +%T) finished $(basename "$f")"
            rm -f "$f"
        fi
    done
    waiting=0
    while IFS='|' read -r name req cmd; do
        [[ -z $name || $name == \#* ]] && continue
        [ -e "$STATE/started/$name" ] && continue
        waiting=$((waiting + 1))
        [ "$running" -ge "$MAX" ] && continue
        req=$(eval echo "$req")
        [ "$req" != "-" ] && [ ! -e "$req" ] && continue
        touch "$STATE/started/$name"
        # shellcheck disable=SC2086
        python $(eval echo "$cmd") > "$STATE/logs/$name.log" 2>&1 < /dev/null &
        echo $! > "$STATE/pids/$name"
        echo "$(date +%T) launched $name (pid $!)"
        running=$((running + 1))
        sleep 60  # stagger data loading
    done < "$JOBS"
    if [ "$waiting" -eq 0 ] && [ "$running" -eq 0 ]; then
        echo "$(date +%T) all jobs in $(basename "$JOBS") finished"
        exit 0
    fi
    sleep 30
done
