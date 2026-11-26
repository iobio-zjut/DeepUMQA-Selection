#!/usr/bin/env bash
# Single-target homo pipeline; called by run.sh with exported paths.
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
    echo "usage: $0 TARGET_NAME (needs G_CSV/MV_CSV/USR_FEATURE_DIR/OUT_DIR)" >&2
    exit 2
fi

# Paths exported by run.sh.
G_CSV="${G_CSV:-}"
MV_CSV="${MV_CSV:-}"
USR_FEATURE_DIR="${USR_FEATURE_DIR:-}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/../output/$target_name}"
SEED="${SEED:-20260226}"

# Hyperparameters.
perc=${perc:-0.25}
w0=${w0:-0.4}   # G weight
w1=${w1:-0.4}   # MV weight
w2=${w2:-0.2}   # USR consensus weight
top_k=${top_k:-500}
chang_k=${chang_k:-15}
max_iter=${max_iter:-8}
group_num=${group_num:-8}

out_target="$OUT_DIR"
usr_feature_dir="$USR_FEATURE_DIR"
usr_consensus_result_dir="$out_target/usr_consensus_result"

step0_csv="$out_target/step0.csv"
step1_csv="$out_target/step1.csv"
step2_csv="$out_target/step2.csv"
step3_csv="$out_target/step3.csv"
all_model="$out_target/all_model.txt"   # from run.sh, .pdb stripped
usr_similarity_matrix="$usr_consensus_result_dir/all_usr_similarity_matrix.npy"
valid_model_names_csv="$usr_consensus_result_dir/all_valid_model_names.csv"
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

echo "[INFO] homo target: $target_name"
echo "       G_CSV=$G_CSV  MV_CSV=$MV_CSV"
echo "       USR=$usr_feature_dir  OUT=$out_target  SEED=$SEED"
require_existing_file "$all_model" "all_model.txt (from run.sh)"
require_existing_file "$G_CSV" "G score CSV"
require_existing_file "$MV_CSV" "MV score CSV"
if [ ! -d "$usr_feature_dir" ]; then
    echo "[ERROR] missing USR feature directory: $usr_feature_dir" >&2
    exit 1
fi

# Steps 1-2: weighted consensus (G/MV/USR) -> step2, plus USR similarity matrix.
if [ ! -f "${step2_csv}" ]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/build_pool.py" \
        --all_model "${all_model}" --g_csv "${G_CSV}" --mv_csv "${MV_CSV}" \
        --perc "${perc}" --w0 "${w0}" --w1 "${w1}" --w2 "${w2}" --top_k "${top_k}" \
        --usr_feature_dir="$usr_feature_dir" --output_dir_usr="$usr_consensus_result_dir" \
        --output_dir="$out_target" --target_name="${target_name}"
    if [ $? -ne 0 ]; then echo "Error: build_pool.py failed"; exit 1; fi
fi

# Step 3: iterative pool refinement -> step3.
if [ ! -f "${step3_csv}" ]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/refine_pool.py" \
        --s_csv "${step2_csv}" --stepn_csv "$step1_csv" \
        --top_k "${top_k}" --chang_k "${chang_k}" --max_iter "${max_iter}" \
        --all_model "${all_model}" --usr_feature_dir="$usr_feature_dir" \
        --output_dir_usr="$usr_consensus_result_dir" --output_dir="$out_target" \
        --target_name="${target_name}"
    if [ $? -ne 0 ]; then echo "Error: refine_pool.py failed"; exit 1; fi
fi

# Step 4: group review -> Select.mass.txt (Top5).
if [ ! -f "${Select_mass_txt}" ]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/groups_review.py" \
        --stepn_csv "$step3_csv" --g_csv "${G_CSV}" --mv_csv "${MV_CSV}" \
        --valid_names_csv "${valid_model_names_csv}" --group_num "${group_num}" \
        --seed "${SEED}" --usr_sm "${usr_similarity_matrix}" \
        --target_name "${target_name}" --out_dir "${out_target}"
    if [ $? -ne 0 ]; then echo "[ERROR] groups_review.py failed"; exit 1; fi
fi

write_pipeline_status "OK" "target=$target_name mode=homo seed=$SEED"
echo "[DONE] Target $target_name finished. Output: $Select_mass_txt"
