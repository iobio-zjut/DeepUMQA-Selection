#!/usr/bin/env bash
# Single-target heter pipeline; called by run.sh. Uses G scores only.
set -u
umask 000

export LANG=C
export LANGUAGE=C
unset LC_ALL

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    else
        PYTHON_BIN=$(command -v python)
    fi
fi

target_name=${1:-}
if [ -z "$target_name" ]; then
    echo "usage: $0 TARGET_NAME (needs G_CSV/USR_FEATURE_DIR/OUT_DIR)" >&2
    exit 2
fi

# Paths exported by run.sh.
G_CSV="${G_CSV:-}"
USR_FEATURE_DIR="${USR_FEATURE_DIR:-}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/../output/$target_name}"

# Hyperparameters.
topk=${topk:-5}
K=${K:-5}
TOTAL_CPUS=${TOTAL_CPUS:-4}

out_target="$OUT_DIR"
usr_feature_dir="$USR_FEATURE_DIR"
usr_consensus_result_dir="$out_target/usr_consensus_result"

all_model="$out_target/all_model.txt"   # from run.sh, .pdb stripped
f_m="$out_target/f_m.txt"
usr_similarity_matrix="$usr_consensus_result_dir/usr_similarity_matrix.npy"
usr_model_names="$usr_consensus_result_dir/valid_models.csv"
Select_mass_txt="$out_target/Select.mass.txt"

state_dir="$out_target/.state"
log_dir="$out_target/logs"
mkdir -p "${out_target}" "${state_dir}" "${log_dir}" "${usr_consensus_result_dir}"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

write_pipeline_status() {
    printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" > "$out_target/pipeline.status"
}

require_existing_file() {
    if [ ! -f "$1" ]; then
        echo "[ERROR] missing $2: $1" >&2
        write_pipeline_status "FAIL" "missing $2: $1"
        exit 1
    fi
}

echo "[INFO] heter target: $target_name"
echo "       G_CSV=$G_CSV  USR=$usr_feature_dir  OUT=$out_target"
require_existing_file "$all_model" "all_model.txt (from run.sh)"
require_existing_file "$G_CSV" "G score CSV"
if [ ! -d "$usr_feature_dir" ]; then
    echo "[ERROR] missing USR feature directory: $usr_feature_dir" >&2
    exit 1
fi

# G anchors + G-threshold pool + USR similarity ranking -> Top5.
if [ ! -f "$f_m" ] || [ ! -f "$usr_similarity_matrix" ] || [ ! -f "$Select_mass_txt" ]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/usr_select.py" \
        --target_name "$target_name" --all_model "$all_model" --G_csv "$G_CSV" \
        --topk "$topk" --K "$K" --usr_feature_dir="$usr_feature_dir" \
        --output_dir "$out_target" --usr_out_dir="$usr_consensus_result_dir" \
        --cpus "${TOTAL_CPUS}"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[ERROR] usr_select.py failed (exit $rc)" >&2
        exit $rc
    fi
    if [ ! -f "$usr_similarity_matrix" ] || [ ! -f "$usr_model_names" ]; then
        echo "[ERROR] similarity matrix or model names not generated" >&2; exit 1
    fi
    if [ ! -f "$Select_mass_txt" ]; then
        echo "[ERROR] Select.mass.txt not generated" >&2; exit 1
    fi
fi

write_pipeline_status "OK" "target=$target_name mode=heter"
echo "[DONE] Target $target_name finished. Output: $Select_mass_txt"
