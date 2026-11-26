#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from os.path import join
import tempfile

def atomic_write_text(path: str, content: str):
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=out_dir, prefix=".tmp.", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write(content)
        tmp_path = handle.name
    os.replace(tmp_path, path)

def atomic_write_dataframe(path: str, dataframe: pd.DataFrame):
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=out_dir, prefix=".tmp.", suffix=".csv", delete=False, encoding="utf-8") as handle:
        dataframe.to_csv(handle, index=False)
        tmp_path = handle.name
    os.replace(tmp_path, path)

# USR feature loading.
def load_usr_features(usr_feature_dir, model_names):
    """Load USR features from one directory (4th array of each npz)."""
    print("\n" + "=" * 80)
    print("[STEP1] Loading USR features")
    print("=" * 80)
    print(f"USR feature dir: {usr_feature_dir}  ({'exists' if os.path.exists(usr_feature_dir) else 'missing'})")

    usr_features = {}
    failed = []
    for name in tqdm(model_names, desc="loading USR features"):
        clean_name = name.rsplit('.pdb', 1)[0] if name.endswith('.pdb') else name
        fpath = os.path.join(usr_feature_dir, clean_name + '.npz')
        if not os.path.exists(fpath):
            failed.append(name)
            continue
        try:
            data = np.load(fpath, allow_pickle=True)
            if len(data.files) <= 3:
                failed.append(name)
                continue
            feat = np.asarray(data[data.files[3]])
            if feat.ndim == 3 and feat.shape[0] == 1:
                feat = feat[0]
            elif feat.ndim != 2:
                failed.append(name)
                continue
            usr_features[name] = feat.astype(np.float64)
        except Exception as e:
            print(f"  [WARN] failed to read {fpath}: {e}")
            failed.append(name)
    print(f"\nLoaded {len(usr_features)}/{len(model_names)} models")
    if failed:
        print(f"Missing features: {len(failed)}")
    return usr_features, failed

# Full-library similarity matrix.
def build_full_similarity_matrix(usr_features, model_names):
    """Compute the pairwise similarity matrix over all models."""
    print("\n" + "=" * 80)
    print("[STEP2] Building full-library similarity matrix")
    print("=" * 80)

    valid_names = [n for n in model_names if n in usr_features]
    n = len(valid_names)
    if n == 0:
        raise RuntimeError("no valid USR features")
    print(f"Valid models: {n}")

    max_len = max(usr_features[name].shape[0] for name in valid_names)
    feat_dim = max_len * 27
    print(f"Max residue length: {max_len}, flattened dim: {feat_dim}")

    X = np.zeros((n, feat_dim), dtype=np.float64)
    for i, name in enumerate(valid_names):
        flat = usr_features[name].reshape(-1)
        if flat.size > feat_dim:
            flat = flat[:feat_dim]
        elif flat.size < feat_dim:
            pad = np.zeros(feat_dim - flat.size, dtype=np.float64)
            flat = np.concatenate([flat, pad])
        X[i] = flat

    print("Computing euclidean distances (cdist)...")
    dist = cdist(X, X, metric='euclidean')
    sim = 1.0 / (1.0 + dist)

    print(f"Similarity matrix shape: {sim.shape}")
    print(f"Similarity range: [{sim.min():.4f}, {sim.max():.4f}]")
    print(f"Mean similarity: {sim.mean():.4f}")
    return sim, valid_names

def main():
    parser = argparse.ArgumentParser(description="two-pool iterative refinement over a full-library similarity matrix")
    parser.add_argument("--s_csv", required=True, help="small pool A model list CSV (name column)")
    parser.add_argument("--stepn_csv", required=True, help="large pool B model list CSV (name column)")
    parser.add_argument("--top_k", type=int, default=500, help="take top_k models from pool A as initial A")
    parser.add_argument("--chang_k", type=int, default=15, help="number of models swapped per round")
    parser.add_argument("--max_iter", type=int, default=8, help="maximum iterations")
    parser.add_argument("--all_model", default="", help="(kept for compatibility, unused)")
    parser.add_argument("--usr_feature_dir", required=True, help="USR feature npz directory")
    parser.add_argument("--output_dir_usr", required=True, help="matrix output directory")
    parser.add_argument("--output_dir", required=True, help="final result output directory")
    parser.add_argument("--target_name", default="", help="(kept for compatibility, unused)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_usr, exist_ok=True)

    # 1. Read pool files.
    print("\n" + "=" * 80)
    print("[STEP0] Reading input files")
    print("=" * 80)
    df_A_full = pd.read_csv(args.s_csv)
    if 'name' not in df_A_full.columns:
        raise ValueError("s_csv must contain a 'name' column")
    if 'score' in df_A_full.columns:
        df_A_full = df_A_full.sort_values('score', ascending=False).reset_index(drop=True)
    init_A = df_A_full.head(args.top_k)['name'].tolist()
    print(f"Initial pool A size: {len(init_A)}")

    df_B = pd.read_csv(args.stepn_csv)
    if 'name' not in df_B.columns:
        raise ValueError("stepn_csv must contain a 'name' column")
    B_names = df_B['name'].tolist()
    print(f"Pool B size: {len(B_names)}")

    # 2. Check final output.
    step3_path = os.path.join(args.output_dir, "step3.csv")
    if os.path.exists(step3_path):
        print(f"\nstep3.csv exists, exiting: {step3_path}")
        df_existing = pd.read_csv(step3_path)
        print(f"Existing model count: {len(df_existing)}")
        print("Top 5 models:")
        print(df_existing.head())
        sys.exit(0)

    all_model_names = list(set(init_A + B_names))
    all_mat_path = os.path.join(args.output_dir_usr, "all_usr_similarity_matrix.npy")
    valid_names_path = os.path.join(args.output_dir_usr, "all_valid_model_names.csv")

    # 3. Load or build similarity matrix.
    if os.path.exists(all_mat_path) and os.path.exists(valid_names_path):
        print("\nFull-library similarity matrix exists, loading")
        sim_matrix = np.load(all_mat_path)
        valid_names = pd.read_csv(valid_names_path)['model_name'].tolist()
        print(f"Valid models: {len(valid_names)}")
    else:
        usr_feat, fail = load_usr_features(args.usr_feature_dir, all_model_names)
        if fail:
            fail_path = os.path.join(args.output_dir_usr, "missing_usr.txt")
            with open(fail_path, 'w') as f:
                f.write('\n'.join(fail))
            print(f"[WARN] missing-feature models written to {fail_path}")
        sim_matrix, valid_names = build_full_similarity_matrix(usr_feat, all_model_names)
        np.save(all_mat_path, sim_matrix.astype(np.float64))
        print(f"Full-library similarity matrix saved to {all_mat_path}")
        pd.DataFrame({"model_name": valid_names}).to_csv(valid_names_path, index=False)
        print(f"Valid name list saved to {valid_names_path}")

    name_to_idx = {name: idx for idx, name in enumerate(valid_names)}

    # Filter out models without valid features.
    init_A = [n for n in init_A if n in name_to_idx]
    B_names = [n for n in B_names if n in name_to_idx]
    if not init_A:
        print("[ERROR] no valid-feature models in pool A, exiting")
        sys.exit(1)
    if not B_names:
        print("[ERROR] no valid-feature models in pool B, exiting")
        sys.exit(1)

    B_indices = np.array([name_to_idx[name] for name in B_names])

    # 4. Iterative refinement.
    print("\n" + "=" * 80)
    print(f"[STEP3] Two-pool iteration ({args.max_iter} rounds, swapping {args.chang_k} each round)")
    print("=" * 80)

    current_A = init_A.copy()
    pre_A_set = set()

    for iter_round in range(1, args.max_iter + 1):
        print(f"\n---------- round {iter_round}/{args.max_iter} ----------")
        cur_set = set(current_A)
        if cur_set == pre_A_set and iter_round > 1:
            print("Pool A unchanged, converged early")
            break
        pre_A_set = cur_set.copy()

        A_indices = np.array([name_to_idx[name] for name in current_A])
        sim_AB = sim_matrix[A_indices][:, B_indices].copy()

        b_col = {name: j for j, name in enumerate(B_names)}
        for i, name in enumerate(current_A):
            j = b_col.get(name)
            if j is not None:
                sim_AB[i, j] = np.nan

        A_scores = np.nanmean(sim_AB, axis=1)
        B_scores = np.nanmean(sim_AB, axis=0)

        A_score_list = list(zip(current_A, A_scores))
        B_score_dict = dict(zip(B_names, B_scores))

        sorted_A = sorted(A_score_list, key=lambda x: x[1], reverse=True)
        drop_num = min(args.chang_k, len(sorted_A))
        drop_set = {name for name, _ in sorted_A[-drop_num:]}
        keep_A = [name for name, _ in sorted_A if name not in drop_set]
        print(f"Dropping bottom {drop_num} models from pool A")

        keep_set = set(keep_A)
        candidates = [b for b in B_names if b not in keep_set and b in B_score_dict]
        sorted_candidates = sorted(candidates, key=lambda x: B_score_dict[x], reverse=True)
        add_list = sorted_candidates[:drop_num]
        print(f"Adding {len(add_list)} models from pool B")

        current_A = keep_A + add_list
        print(f"Pool A size after this round: {len(current_A)}")

    # 5. Generate step3.csv.
    print("\n" + "=" * 80)
    print("[STEP4] Writing final result step3.csv")
    print("=" * 80)

    final_A_indices = np.array([name_to_idx[name] for name in current_A])
    final_sim_AB = sim_matrix[final_A_indices][:, B_indices]
    final_scores = final_sim_AB.mean(axis=1)

    df_step3 = pd.DataFrame({
        'name': current_A,
        'score': final_scores
    }).sort_values('score', ascending=False).reset_index(drop=True)

    atomic_write_dataframe(step3_path, df_step3)
    print(f"step3.csv generated with {len(df_step3)} models")


if __name__ == "__main__":
    main()
