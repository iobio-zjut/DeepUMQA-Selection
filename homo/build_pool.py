#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from os.path import join, exists


def normalize_model_name(filename):
    """Strip path, .pdb suffix and ranked prefixes; return the clean name."""
    name = os.path.basename(str(filename).strip()).replace(".pdb", "")
    name = name.replace(".pdb", "")
    name = re.sub(r'^ranked_\d+_', '', name)
    name = re.sub(r'_ranked_\d+_unrelaxed', '', name)
    return name


def load_all_model(filepath):
    """Load all_model.txt; return raw names, clean set, and clean->raw map."""
    if not exists(filepath):
        raise FileNotFoundError(f"all_model file not found: {filepath}")
    with open(filepath, 'r') as f:
        raw_names = [line.strip() for line in f if line.strip()]

    clean2raw = {}
    clean_set = set()
    for rn in raw_names:
        cn = normalize_model_name(rn)
        if cn in clean2raw:
            print(f"[WARN] duplicate clean name: {cn} -> {clean2raw[cn]}, {rn}; keeping first")
        else:
            clean2raw[cn] = rn
        clean_set.add(cn)

    print(f"[ALL_MODEL] raw names: {len(raw_names)}, unique clean names: {len(clean_set)}")
    return raw_names, clean_set, clean2raw


def process_score_csv(csv_path, score_col_name, clean2raw):
    """Load a score CSV and return a DataFrame of raw names and scores."""
    df = pd.read_csv(csv_path)
    if "name" not in df.columns or "score" not in df.columns:
        raise KeyError(f"{csv_path} must contain 'name' and 'score' columns")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    if df.empty:
        raise ValueError(f"{csv_path} has no valid score data")

    records = []
    fail_count = 0
    for _, row in df.iterrows():
        raw_name_csv = str(row["name"]).strip()
        score = float(row["score"])
        clean_name = normalize_model_name(raw_name_csv)
        if clean_name in clean2raw:
            orig_name = clean2raw[clean_name]
            records.append({"raw_name": orig_name, score_col_name: score})
        else:
            fail_count += 1

    print(f"[{csv_path}] rows: {len(df)}, matched: {len(records)}, unmatched: {fail_count}")
    if not records:
        raise RuntimeError(f"{csv_path} has no model matching the all_model whitelist")
    df_out = pd.DataFrame(records)
    df_out = df_out.sort_values([score_col_name], ascending=False).drop_duplicates(subset="raw_name", keep="first")
    print(f"[{csv_path}] unique valid models: {len(df_out)}")
    return df_out


def load_usr_features(usr_feature_dir, model_names):
    """Load USR features (.npz); model_names holds raw names."""
    if not exists(usr_feature_dir):
        raise FileNotFoundError(f"USR feature directory not found: {usr_feature_dir}")
    usr_features = {}
    failed = []
    for name in tqdm(model_names, desc="loading USR features"):
        feat_file = join(usr_feature_dir, name + ".npz")
        if not exists(feat_file):
            failed.append(name)
            continue
        try:
            data = np.load(feat_file, allow_pickle=True)
            if len(data.files) <= 3:
                failed.append(name)
                continue
            feat = np.asarray(data[data.files[3]])
            if feat.ndim == 3 and feat.shape[0] == 1:
                feat = feat[0]
            elif feat.ndim != 2:
                failed.append(name)
                continue
            usr_features[name] = feat
        except Exception as e:
            failed.append(name)
    print(f"Loaded {len(usr_features)}/{len(model_names)} USR features")
    if failed:
        print(f"Missing features: {len(failed)}")
    return usr_features


def compute_usr_matrix(usr_features, model_names, output_dir_usr):
    """Compute the similarity matrix; save .npy and the valid-name list."""
    valid_names = [n for n in model_names if n in usr_features]
    n = len(valid_names)
    if n == 0:
        raise RuntimeError("no valid USR features to compute matrix")
    print(f"Valid USR models: {n}")

    max_len = max(usr_features[name].shape[0] for name in valid_names)
    feat_dim = max_len * 27
    feature_matrix = np.zeros((n, feat_dim), dtype=np.float32)
    for i, name in enumerate(valid_names):
        flat = usr_features[name].reshape(-1)
        if flat.size > feat_dim:
            flat = flat[:feat_dim]
        elif flat.size < feat_dim:
            flat = np.concatenate([flat, np.zeros(feat_dim - flat.size, dtype=flat.dtype)])
        feature_matrix[i] = flat

    distance_matrix = cdist(feature_matrix, feature_matrix, metric='euclidean')
    similarity_matrix = 1.0 / (1.0 + distance_matrix)

    os.makedirs(output_dir_usr, exist_ok=True)
    np.save(join(output_dir_usr, "usr_similarity_matrix.npy"), similarity_matrix)
    print(f"Similarity matrix saved: {join(output_dir_usr, 'usr_similarity_matrix.npy')}")

    valid_df = pd.DataFrame({"model_name": valid_names})
    valid_df.to_csv(join(output_dir_usr, "valid_model_names.csv"), index=False)
    print(f"Valid name list saved: {join(output_dir_usr, 'valid_model_names.csv')}")
    return similarity_matrix, valid_names


def compute_avg_usr_scores(similarity_matrix, valid_names):
    """Mean similarity of each model to the others (self excluded)."""
    scores = []
    for i in range(len(valid_names)):
        row = similarity_matrix[i]
        avg = np.mean(np.delete(row, i))
        scores.append(avg)
    return dict(zip(valid_names, scores))


def minmax_norm(series: pd.Series) -> pd.Series:
    """MinMax normalization to [0, 1]."""
    min_val = series.min()
    max_val = series.max()
    if min_val == max_val:
        return pd.Series(np.zeros_like(series.values), index=series.index)
    return (series - min_val) / (max_val - min_val)


def safe_load_csv_with_scores(path, score_cols):
    """Load a CSV, coerce score columns to numeric, drop invalid rows."""
    df = pd.read_csv(path)
    for col in score_cols:
        if col not in df.columns:
            raise KeyError(f"missing column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=score_cols)
    if df.empty:
        raise ValueError(f"{path} has no valid data after cleaning (non-numeric scores)")
    return df


def main():
    parser = argparse.ArgumentParser(description="G/MV intersection filter + weighted USR consensus")
    parser.add_argument("--all_model", required=True, help="model whitelist file (extensionless names, one per line)")
    parser.add_argument("--g_csv", required=True, help="G score table (name,score)")
    parser.add_argument("--mv_csv", required=True, help="MV score table (name,score)")
    parser.add_argument("--perc", type=float, default=0.25, help="top fraction to take (0~1), default 0.25")
    parser.add_argument("--w0", type=float, default=0.4, help="G weight, default 0.4")
    parser.add_argument("--w1", type=float, default=0.4, help="MV weight, default 0.4")
    parser.add_argument("--w2", type=float, default=0.2, help="USR weight, default 0.2")
    parser.add_argument("--top_k", type=int, default=500, help="number of top models to keep")
    parser.add_argument("--usr_feature_dir", required=True, help="USR feature directory")
    parser.add_argument("--output_dir_usr", required=True, help="USR matrix output directory")
    parser.add_argument("--output_dir", required=True, help="final result output directory")
    parser.add_argument("--target_name", default="unknown", help="target name (for logging)")
    args = parser.parse_args()

    if not (0 < args.perc <= 1):
        print("[ERROR] --perc must be in (0, 1]")
        sys.exit(1)
    for path_flag, path in [("all_model", args.all_model), ("g_csv", args.g_csv),
                            ("mv_csv", args.mv_csv), ("usr_feature_dir", args.usr_feature_dir)]:
        if not exists(path):
            print(f"[ERROR] {path_flag} path not found -> {path}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_usr, exist_ok=True)

    # 1. Load whitelist mapping.
    raw_names, clean_set, clean2raw = load_all_model(args.all_model)

    # 2. Process G/MV score tables.
    try:
        df_g = process_score_csv(args.g_csv, "g_score", clean2raw)
    except (ValueError, RuntimeError, KeyError) as e:
        print(f"[ERROR] failed to process {args.g_csv}: {e}")
        sys.exit(1)

    try:
        df_mv = process_score_csv(args.mv_csv, "mv_score", clean2raw)
    except (ValueError, RuntimeError, KeyError) as e:
        print(f"[ERROR] failed to process {args.mv_csv}: {e}")
        sys.exit(1)

    # ---- step0.csv ----
    step0_path = join(args.output_dir, "step0.csv")
    if exists(step0_path):
        print(f"step0.csv exists, loading: {step0_path}")
        try:
            inter_df = safe_load_csv_with_scores(step0_path, ["g_score", "mv_score"])
        except Exception as e:
            print(f"[ERROR] failed to load step0.csv: {e}")
            sys.exit(1)

        if "name" not in inter_df.columns:
            print("[ERROR] step0.csv missing 'name' column")
            sys.exit(1)
        inter_df.rename(columns={"name": "raw_name"}, inplace=True)
    else:
        perc = args.perc
        g_top_n = max(1, int(len(df_g) * perc))
        mv_top_n = max(1, int(len(df_mv) * perc))
        df_g_top = df_g.sort_values("g_score", ascending=False).head(g_top_n)
        df_mv_top = df_mv.sort_values("mv_score", ascending=False).head(mv_top_n)

        inter_df = pd.merge(df_g_top, df_mv_top, on="raw_name", how="inner")
        if inter_df.empty:
            print(f"[ERROR] G/MV top {perc*100:.0f}% intersection is empty")
            sys.exit(1)
        print(f"G top {perc*100:.0f}%: {g_top_n}, MV top {perc*100:.0f}%: {mv_top_n}, intersection: {len(inter_df)}")

        step0_out = inter_df[["raw_name", "g_score", "mv_score"]].copy()
        step0_out.rename(columns={"raw_name": "name"}, inplace=True)
        step0_out.to_csv(step0_path, index=False, float_format="%.8f")
        print(f"step0.csv saved (raw names): {step0_path}")

    # ---- step1.csv ----
    step1_path = join(args.output_dir, "step1.csv")
    if exists(step1_path):
        print(f"step1.csv exists, loading: {step1_path}")
        try:
            df_g_median = safe_load_csv_with_scores(step1_path, ["score"])
        except Exception as e:
            print(f"[ERROR] failed to load step1.csv: {e}")
            sys.exit(1)

        if "name" not in df_g_median.columns:
            print("[ERROR] step1.csv missing 'name' column")
            sys.exit(1)
        print(f"Loaded step1.csv with {len(df_g_median)} models")
    else:
        g_median = df_g["g_score"].median()
        df_g_median = df_g[df_g["g_score"] >= g_median].copy()
        step1_out = df_g_median[["raw_name", "g_score"]].copy()
        step1_out.rename(columns={"raw_name": "name", "g_score": "score"}, inplace=True)
        step1_out.to_csv(step1_path, index=False, float_format="%.8f")
        print(f"step1.csv saved (raw names, median={g_median:.8f}): {step1_path}")

    # ---- USR matrix ----
    usr_sim_path = join(args.output_dir_usr, "usr_similarity_matrix.npy")
    valid_csv_path = join(args.output_dir_usr, "valid_model_names.csv")
    inter_model_names = inter_df["raw_name"].tolist()

    if exists(usr_sim_path) and exists(valid_csv_path):
        print(f"USR similarity matrix exists, loading: {usr_sim_path}")
        sim_matrix = np.load(usr_sim_path)
        valid_df = pd.read_csv(valid_csv_path)
        valid_names = valid_df["model_name"].tolist()
    else:
        usr_features = load_usr_features(args.usr_feature_dir, inter_model_names)
        if not usr_features:
            print("[ERROR] no USR features available, aborting USR analysis")
            sys.exit(1)
        sim_matrix, valid_names = compute_usr_matrix(usr_features, inter_model_names, args.output_dir_usr)

    # ---- USR mean score (only intersection models with USR features) ----
    usr_scores = compute_avg_usr_scores(sim_matrix, valid_names)

    inter_df_indexed = inter_df.set_index("raw_name")
    common_names = [name for name in inter_df_indexed.index if name in usr_scores]
    if not common_names:
        print("[ERROR] no intersection model has USR features")
        sys.exit(1)

    g_scores = inter_df_indexed.loc[common_names, "g_score"].values
    mv_scores = inter_df_indexed.loc[common_names, "mv_score"].values
    usr_scores_list = [usr_scores[name] for name in common_names]

    merged_df = pd.DataFrame({
        "name": common_names,
        "g_score": g_scores,
        "mv_score": mv_scores,
        "usr_score": usr_scores_list
    })

    merged_df["norm_g"] = minmax_norm(merged_df["g_score"])
    merged_df["norm_mv"] = minmax_norm(merged_df["mv_score"])
    merged_df["norm_usr"] = minmax_norm(merged_df["usr_score"])

    w0, w1, w2 = args.w0, args.w1, args.w2
    merged_df["score"] = (w0 * merged_df["norm_g"] +
                          w1 * merged_df["norm_mv"] +
                          w2 * merged_df["norm_usr"])

    merged_df = merged_df.sort_values("score", ascending=False)
    if args.top_k and len(merged_df) > args.top_k:
        merged_df = merged_df.head(args.top_k)

    # ---- step2.csv ----
    step2_path = join(args.output_dir, "step2.csv")
    if exists(step2_path):
        print(f"step2.csv exists, skipping: {step2_path}")
    else:
        merged_df.to_csv(step2_path, index=False, float_format="%.8f")
        print(f"step2.csv saved (raw names, top {len(merged_df)}): {step2_path}")


if __name__ == "__main__":
    main()
