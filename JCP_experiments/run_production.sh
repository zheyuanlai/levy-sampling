#!/usr/bin/env bash
# ==========================================================================
# LSC-CP JCP production launcher.  DO NOT run unattended without checking the
# GPU is free -- this executes the full E1-E4 x method-matrix x all-seed runs
# (E3: N=2000,T=200; E4: N=1000,T=100), which take hours.
#
# Each experiment is gated by the GENEROUS-box weak-stationarity certificate:
# if max R >= 1e-6 the experiment is REFUSED (fix quadrature/box first).
# Notebooks run dt-refinement and quadrature-refinement themselves before the
# production run.
#
# Method matrix (plan): E1/E2 dual-run exact + RA; E3/E4 run RA (exact
# quadrature is prohibitively expensive there: q_theta*A*q_rho ~ 1024-1536 V
# per particle per step vs q_theta for RA).
#
# Usage:   JCP_GPU=5 ./run_production.sh            # GPU must be one of 4-7
#          JCP_GPU=4 JCP_REGEN=0 ./run_production.sh  # skip notebook regen
# ==========================================================================
set -euo pipefail

JCP_GPU="${JCP_GPU:-4}"
# default allowed 4-7; a specific GPU can be opted-in via JCP_EXTRA_GPUS (only
# when verified free and belonging to your group -- GPUs 0-3 default off-limits)
case ",4,5,6,7,${JCP_EXTRA_GPUS:-}," in
    *",$JCP_GPU,"*) ;;
    *) echo "ERROR: JCP_GPU='$JCP_GPU' not allowed (default 4-7; opt-in via JCP_EXTRA_GPUS). GPUs 0-3 default off-limits." >&2
       exit 2 ;;
esac
export JCP_GPU JCP_EXTRA_GPUS
JCP_REGEN="${JCP_REGEN:-1}"

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "=================================================================="
echo " LSC-CP production run   GPU=$JCP_GPU   $(date)"
echo "=================================================================="

if [ "$JCP_REGEN" = "1" ]; then
    echo "== regenerating notebooks from notebooks/build_notebooks.py =="
    python notebooks/build_notebooks.py
fi

# method matrices
DUAL="ULA,MALA,FLA,BAOAB,PT,CP,LSC-CP,CP-RA,LSC-CP-RA"    # E1/E2: exact + RA
RA="ULA,MALA,FLA,BAOAB,PT,CP-RA,LSC-CP-RA"                # E3/E4: RA

# experiment order: name  notebook  methods
run_one () {
    local name="$1" nb="$2" methods="$3"
    echo ""
    echo "------------------------------------------------------------------"
    echo "  $name : certificate pre-flight gate"
    echo "------------------------------------------------------------------"
    if python scripts/certificate_gate.py "$name"; then
        echo "  $name : gate PASSED -> running $nb"
        echo "  methods = $methods"
        JCP_METHODS="$methods" python notebooks/run_notebook.py "notebooks/$nb"
    else
        echo "  $name : certificate gate FAILED -> REFUSING to run (fix quadrature/box first)"
    fi
}

run_one double_well  01_double_well.ipynb  "$DUAL"
run_one mog40        02_mog40.ipynb        "$DUAL"
run_one mb3well_10d  03_mb3well_10d.ipynb  "$RA"
run_one coupled_phi4 04_coupled_phi4.ipynb "$RA"

echo ""
echo "== production launcher done =="
