#!/usr/bin/env bash
# Host-side aux-server launcher. One round only; never chains another sweep.
set -euo pipefail
cd /home/wll/dev/research/sralt
mkdir -p runs/smia_tuning/round1
run_worker() {
  local dataset="$1" gpu="$2"
  /home/wll/bin/apptainer exec --nv --containall --cleanenv \
    --home /home/wll/container/ubuntu-home:/home/wll \
    --bind /home/wll/dev:/home/wll/dev --bind /home/wll/container/tmp:/tmp \
    --env CUDA_VISIBLE_DEVICES="$gpu",OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2 \
    /home/wll/container/ubuntu /bin/bash -c \
    "cd /home/wll/dev/research/sralt && .venv/bin/python -u scripts/tune_smia.py --datasets $dataset --output runs/smia_tuning/round1/$dataset" \
    >"runs/smia_tuning/round1/$dataset-console.log" 2>&1
}
run_worker ciao 0 &
ciao_pid=$!
run_worker flickr 1 &
flickr_pid=$!
while kill -0 "$ciao_pid" 2>/dev/null || kill -0 "$flickr_pid" 2>/dev/null; do
  python3 scripts/analyze_smia_round.py runs/smia_tuning/round1
  sleep 30
done
exit_code=0
wait "$ciao_pid" || exit_code=$?
wait "$flickr_pid" || exit_code=$?
python3 scripts/analyze_smia_round.py runs/smia_tuning/round1
if [ "$exit_code" -ne 0 ]; then
  printf '%s\n' "Worker failed (exit $exit_code); inspect worker logs. No further round started."
  exit "$exit_code"
fi
printf '%s\n' 'Round 1 finished. Await user confirmation before continuing.'
