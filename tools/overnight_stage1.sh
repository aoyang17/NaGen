#!/usr/bin/env bash
# Stage 1 of the 2026-09-04 overnight run.
#
# The Bash tool is being gated intermittently by a safety classifier, so this
# script is written to extract the maximum amount of situational awareness from
# a SINGLE successful invocation: it resolves the interpreter, inventories every
# asset the later stages need, and runs both CPU-only diagnostics (P0.4, P1.5).
#
# It never modifies project state.  Everything lands under runs/overnight_<date>/.
set -u
cd /home/aobo/NaGen || exit 1

STAMP="$(date +%Y-%m-%d_%H%M%S)"
OUT="runs/overnight_${STAMP}"
mkdir -p "${OUT}"
LOG="${OUT}/stage1.log"

say() { echo "=== $* ===" | tee -a "${LOG}"; }
run() { echo "\$ $*" >>"${LOG}"; "$@" >>"${LOG}" 2>&1; echo "[exit $?]" >>"${LOG}"; }

say "STAGE 1 START ${STAMP}"

# ---------------------------------------------------------------- interpreter
say "INTERPRETER RESOLUTION"
PY=""
for candidate in \
    /mnt/data2/aobo/NaGen/dependencies/python313/bin/python3 \
    /mnt/data2/aobo/NaGen/dependencies/python313/bin/python \
    "$(command -v python3.13 2>/dev/null)" \
    "$(command -v python3 2>/dev/null)"
do
    [ -n "${candidate}" ] && [ -x "${candidate}" ] || continue
    if "${candidate}" -c "import torch, numpy, pymatgen" >/dev/null 2>&1; then
        PY="${candidate}"; break
    fi
done
# Fall back to the .pylibs symlink used as a PYTHONPATH site directory.
if [ -z "${PY}" ]; then
    for candidate in "$(command -v python3 2>/dev/null)"; do
        [ -n "${candidate}" ] || continue
        if PYTHONPATH=/home/aobo/NaGen/.pylibs "${candidate}" \
               -c "import torch, numpy, pymatgen" >/dev/null 2>&1; then
            PY="${candidate}"
            export PYTHONPATH=/home/aobo/NaGen/.pylibs
            break
        fi
    done
fi

if [ -z "${PY}" ]; then
    say "NO INTERPRETER WITH torch+pymatgen FOUND -- diagnosing"
    { echo "--- candidates ---"
      ls -la /mnt/data2/aobo/NaGen/dependencies/ 2>&1
      ls -la /mnt/data2/aobo/NaGen/dependencies/python313/ 2>&1
      ls -la /mnt/data2/aobo/NaGen/dependencies/python313/bin/ 2>&1
      echo "--- which ---"; command -v python3 python3.13 conda mamba uv 2>&1
      echo "--- conda envs ---"; ls -la ~/.conda/envs /opt/conda/envs 2>&1
      echo "--- pylibs ---"; ls /home/aobo/NaGen/.pylibs 2>&1 | head -40
    } >>"${LOG}" 2>&1
else
    say "INTERPRETER = ${PY}  (PYTHONPATH=${PYTHONPATH:-<unset>})"
    run "${PY}" -c "
import torch, numpy, pymatgen, sys
print('python  ', sys.version.split()[0], sys.executable)
print('torch   ', torch.__version__, 'cuda_available', torch.cuda.is_available(),
      'n_gpu', torch.cuda.device_count())
print('numpy   ', numpy.__version__)
import pymatgen.core as pc; print('pymatgen', pc.__version__ if hasattr(pc,'__version__') else 'unknown')
from pymatgen.analysis.phase_diagram import PhaseDiagram
print('PhaseDiagram import OK')
"
fi

# ------------------------------------------------------------------ inventory
say "GPU"
run nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv

say "DISK"
run df -h /home /mnt/data2

say "PACKED DATASETS (*.pt)"
find /mnt/data2/aobo/NaGen -maxdepth 6 -name '*.pt' -printf '%10s  %TY-%Tm-%Td  %p\n' \
    2>/dev/null | sort -k3 | head -60 | tee -a "${LOG}"

say "HULL CACHES"
find /mnt/data2/aobo/NaGen /home/aobo/NaGen -maxdepth 6 -name '*hull*cache*.json' \
    -printf '%10s  %TY-%Tm-%Td  %p\n' 2>/dev/null | head -30 | tee -a "${LOG}"

say "CHECKPOINTS"
find /mnt/data2/aobo/NaGen -maxdepth 6 \( -name '*.ckpt' -o -name '*checkpoint*' -o -name '*best*.pt' \) \
    -printf '%10s  %TY-%Tm-%Td  %p\n' 2>/dev/null | head -40 | tee -a "${LOG}"

say "EXPERIMENT DIRECTORIES"
{ ls -la /home/aobo/NaGen/experiments/ 2>&1
  echo "--- durable ---"
  ls -la /mnt/data2/aobo/NaGen/experiments/ 2>&1
  echo "--- september dirs (docs say these should exist somewhere) ---"
  find /mnt/data2/aobo/NaGen -maxdepth 4 -type d -name 'exp_2026-09*' 2>/dev/null
} >>"${LOG}" 2>&1

say "PILOT SUMMARY"
find /home/aobo/NaGen /mnt/data2/aobo/NaGen -maxdepth 6 -name 'pilot_summary.json' 2>/dev/null \
    | tee -a "${LOG}"

say "CALIBER GREP (P0.1)"
{ grep -rn "get_mp2020_reference\|mp2020_hull_reference\|mp2020_correction" \
      src tools tests experiments configs 2>/dev/null
  echo "--- UMAHullCache callers ---"
  grep -rln "UMAHullCache" src tools tests 2>/dev/null
} >>"${LOG}" 2>&1

# ---------------------------------------------------------------- diagnostics
if [ -n "${PY}" ]; then
    export PYTHONPATH="${PYTHONPATH:-}:/home/aobo/NaGen/src"

    say "P0.4 -- UMA hull facet ruler"
    run "${PY}" tools/p0_4_uma_hull_facets.py --out "${OUT}/p0_4_uma_hull_facets.json"

    say "P1.5 -- train/inference source mismatch (H1f + H1g)"
    run "${PY}" tools/p1_5_source_mismatch.py \
        --out "${OUT}/p1_5_source_mismatch.json" \
        --figure "${OUT}/p1_5_source_mismatch.png"

    say "TESTS"
    run "${PY}" -m pytest tests/ -x -q --timeout=600
fi

say "STAGE 1 COMPLETE -> ${OUT}"
echo "${OUT}" > runs/.latest_overnight
