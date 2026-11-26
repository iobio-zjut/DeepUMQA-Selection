#!/usr/bin/env bash
# Single-target consensus QA entry: run.sh <target> <homo|heter> [seed] --model-list FILE [--usr-feat-dir DIR]
set -u
umask 000

export LANG=C
export LANGUAGE=C
unset LC_ALL

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# Resolve Python from the active environment.
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    else
        PYTHON_BIN=$(command -v python)
    fi
fi

# Optional thread limits from SLURM or TOTAL_CPUS.
TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-${TOTAL_CPUS:-4}}"
export TOTAL_CPUS
if [ -n "${TOTAL_CPUS}" ]; then
    export OMP_NUM_THREADS="${TOTAL_CPUS}"
    export MKL_NUM_THREADS="${TOTAL_CPUS}"
    export OPENBLAS_NUM_THREADS="${TOTAL_CPUS}"
    export NUMEXPR_NUM_THREADS="${TOTAL_CPUS}"
fi

usage() {
    cat <<'EOF'
Usage:
  run.sh <target> <homo|heter> [seed] --model-list FILE [--usr-feat-dir DIR]

Arguments:
  target          e.g. T1292 / H1208
  mode            homo or heter (explicit, never inferred from the name)
  seed            grouping seed for group review, homo only (default: 20260226)

Options:
  --model-list FILE   one full PDB path per line (multiple dirs allowed, same target)
  --usr-feat-dir DIR  external dir of .npz features; built locally if omitted
  -h, --help          show this help

Input CSV (fixed names):
  DeepUMQA-Global/<target>.csv   G scores (homo and heter)
  MViewEMA/<target>.csv          MV scores (homo only)

Output: output/<target>/
EOF
}

# --- parse arguments ---
POSITIONAL=()
MODEL_LIST=""
USR_FEAT_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model-list)   MODEL_LIST="${2:-}"; shift 2 ;;
        --usr-feat-dir) USR_FEAT_DIR="${2:-}"; shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        -*)             echo "[ERROR] unknown option: $1" >&2; usage; exit 2 ;;
        *)              POSITIONAL+=("$1"); shift ;;
    esac
done

target="${POSITIONAL[0]:-}"
mode="${POSITIONAL[1]:-}"
seed="${POSITIONAL[2]:-20260226}"

if [ -z "${target}" ] || [ -z "${mode}" ]; then
    echo "[ERROR] target and mode are required" >&2; usage; exit 2
fi
if [ "${mode}" != "homo" ] && [ "${mode}" != "heter" ]; then
    echo "[ERROR] mode must be homo or heter, got: ${mode}" >&2; exit 2
fi
if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] seed must be an integer, got: ${seed}" >&2; exit 2
fi
if [ -z "${MODEL_LIST}" ] || [ ! -f "${MODEL_LIST}" ]; then
    echo "[ERROR] --model-list not found: ${MODEL_LIST:-<missing>}" >&2; exit 2
fi
if [ -n "${USR_FEAT_DIR}" ] && [ ! -d "${USR_FEAT_DIR}" ]; then
    echo "[ERROR] --usr-feat-dir not found: ${USR_FEAT_DIR}" >&2; exit 2
fi

# --- paths derived from SCRIPT_DIR ---
G_CSV="$SCRIPT_DIR/DeepUMQA-Global/${target}.csv"
MV_CSV="$SCRIPT_DIR/MViewEMA/${target}.csv"
OUT_DIR="$SCRIPT_DIR/output/${target}"
LOCAL_USR_ROOT="$SCRIPT_DIR/USR"
LOCAL_USR_DIR="$LOCAL_USR_ROOT/${target}"
MODEL_LIST_OUT="$OUT_DIR/model_list.txt"
ALL_MODEL_OUT="$OUT_DIR/all_model.txt"

mkdir -p "$OUT_DIR"

echo "========================================================================"
echo " Single-target consensus QA"
echo "   target       : $target"
echo "   mode         : $mode"
echo "   seed         : $seed  (homo only)"
echo "   model-list   : $MODEL_LIST"
echo "   usr-feat-dir : ${USR_FEAT_DIR:-<local build -> $LOCAL_USR_DIR>}"
echo "   output       : $OUT_DIR"
echo "========================================================================"

# --- 1. validate input CSVs ---
echo "[STEP 1] Validating input CSVs"
"$PYTHON_BIN" - "$G_CSV" "$MV_CSV" "$mode" <<'PY'
import sys
import pandas as pd

g_path, mv_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]

def check(path):
    df = pd.read_csv(path)
    cols = list(df.columns)
    if "name" not in cols or "score" not in cols:
        raise SystemExit("[ERROR] {} needs name,score columns (got: {})".format(path, cols))
    names = df["name"].astype(str).tolist()
    bad = [n for n in names if not n.lower().endswith(".pdb")]
    if bad:
        raise SystemExit("[ERROR] {} name must end with .pdb, e.g. {}".format(path, bad[:3]))
    try:
        df["score"].astype(float)
    except Exception as e:
        raise SystemExit("[ERROR] {} score has non-numeric values: {}".format(path, e))
    return names

g_names = check(g_path)
if mode == "homo":
    mv_names = check(mv_path)
    if set(g_names) != set(mv_names):
        only_g = list(set(g_names) - set(mv_names))[:3]
        only_mv = list(set(mv_names) - set(g_names))[:3]
        raise SystemExit("[ERROR] G/MV name sets differ; G-only: {}; MV-only: {}".format(only_g, only_mv))
    print("[OK] CSVs valid: G={}, MV={}, name sets match".format(len(g_names), len(mv_names)))
else:
    print("[OK] CSV valid: G={} (heter uses G only)".format(len(g_names)))
PY
if [ $? -ne 0 ]; then
    echo "[ERROR] CSV validation failed" >&2; exit 2
fi

# --- 2. write model_list.txt and all_model.txt ---
echo "[STEP 2] Writing model_list.txt and all_model.txt"
cp "$MODEL_LIST" "$MODEL_LIST_OUT"

"$PYTHON_BIN" - "$G_CSV" "$ALL_MODEL_OUT" <<'PY'
import sys, os
import pandas as pd

g_path, out_path = sys.argv[1], sys.argv[2]
df = pd.read_csv(g_path)
names = [os.path.basename(str(x).strip()) for x in df["name"]]
names = [n[:-4] if n.lower().endswith(".pdb") else n for n in names]  # strip .pdb only
with open(out_path, "w") as f:
    f.write("\n".join(names) + "\n")
print("[OK] all_model.txt: {} rows".format(len(names)))
PY
if [ $? -ne 0 ]; then
    echo "[ERROR] failed to write model lists" >&2; exit 1
fi

n_models=$(wc -l < "$ALL_MODEL_OUT" | tr -d ' ')
echo "       model_list.txt -> $MODEL_LIST_OUT"
echo "       all_model.txt  -> $ALL_MODEL_OUT ($n_models rows)"

# --- 3. USR features: external first, else local build ---
echo "[STEP 3] Preparing USR features"
if [ -n "${USR_FEAT_DIR}" ]; then
    ACTUAL_USR_DIR="$USR_FEAT_DIR"
    echo "[INFO] using external USR features: $ACTUAL_USR_DIR"
else
    ACTUAL_USR_DIR="$LOCAL_USR_DIR"
    n_npz=$(find "$ACTUAL_USR_DIR" -maxdepth 1 -type f -name "*.npz" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n_npz" -ge "$n_models" ] && [ "$n_models" -gt 0 ]; then
        echo "[SKIP] local USR features complete ($n_npz >= $n_models): $ACTUAL_USR_DIR"
    else
        echo "[INFO] building USR features -> $ACTUAL_USR_DIR ($n_npz/$n_models present)"
        "$PYTHON_BIN" "$SCRIPT_DIR/USR_npz.py" \
            --save_dir "$LOCAL_USR_ROOT" \
            --model_list_path "$MODEL_LIST_OUT" \
            --target_name "$target"
        if [ $? -ne 0 ]; then
            echo "[ERROR] USR_npz.py failed" >&2; exit 1
        fi
        n_npz=$(find "$ACTUAL_USR_DIR" -maxdepth 1 -type f -name "*.npz" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$n_npz" -lt "$n_models" ]; then
            echo "[ERROR] USR feature count short: got $n_npz, expected $n_models" >&2; exit 1
        fi
        echo "[OK] USR features built: $n_npz npz files"
    fi
fi

# --- 4. dispatch to the mode-specific pipeline ---
echo "[STEP 4] Running $mode pipeline"
export G_CSV="$G_CSV"
export MV_CSV="$MV_CSV"
export USR_FEATURE_DIR="$ACTUAL_USR_DIR"
export OUT_DIR="$OUT_DIR"
export SEED="$seed"

if [ "$mode" = "homo" ]; then
    bash "$SCRIPT_DIR/homo/homo_select.sh" "$target"
else
    bash "$SCRIPT_DIR/heter/heter_select.sh" "$target"
fi
rc=$?
if [ $rc -ne 0 ]; then
    echo "[ERROR] $mode pipeline failed (exit $rc)" >&2; exit $rc
fi

echo "========================================================================"
echo "[DONE] $target ($mode) finished"
echo "       Final result: $OUT_DIR/Select.mass.txt"
echo "========================================================================"
