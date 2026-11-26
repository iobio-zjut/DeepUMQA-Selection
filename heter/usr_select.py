#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
import concurrent.futures


def normalize_massive_pred_name(filename: str) -> str:
    """Normalize a model name to '<name>.pdb' by stripping ranked prefixes."""
    name = os.path.basename(str(filename).strip())
    if name.lower().endswith('.pdb'):
        name = name[:-4]
    name = re.sub(r'^ranked_\d+_', '', name)
    name = re.sub(r'_ranked_\d+_unrelaxed', '', name)
    name = name + ".pdb"
    return name


def load_usr_features_single(usr_feature_dir, model_names, num_workers=4):
    """Load USR features (.npz) in parallel; return {model: feature_array (L,27)}."""
    print("\n" + "=" * 80)
    print(f"[Loading USR features] ({num_workers} threads)")
    print("Feature dir: {}".format(usr_feature_dir))
    print("=" * 80)

    if not os.path.isdir(usr_feature_dir):
        print("[ERROR] feature directory not found -> {}".format(usr_feature_dir))
        sys.exit(1)

    tasks = []
    for name in model_names:
        path = os.path.join(usr_feature_dir, name + '.npz')
        if os.path.exists(path):
            tasks.append((name, path))

    usr_features = {}
    failed_models = set(model_names) - set(t[0] for t in tasks)

    def load_one(name, path):
        try:
            data = np.load(path, allow_pickle=True)
            if len(data.files) == 0 or len(data.files) <= 3:
                return None, name
            fourth_key = data.files[3]
            feat = np.asarray(data[fourth_key])
            if feat.ndim == 3 and feat.shape[0] == 1:
                feat = feat[0]
            elif feat.ndim == 2:
                pass
            else:
                return None, name
            return feat, name
        except Exception as e:
            print(f"  [WARN] failed to read {path}: {e}")
            return None, name

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_name = {executor.submit(load_one, name, path): name for name, path in tasks}
        for future in tqdm(concurrent.futures.as_completed(future_to_name),
                           total=len(tasks), desc="loading USR features"):
            feat, name = future.result()
            if feat is not None:
                usr_features[name] = feat
            else:
                failed_models.add(name)

    print("Loaded features for {}/{} models".format(len(usr_features), len(model_names)))
    if failed_models:
        print("Missing/failed: {} models".format(len(failed_models)))
    return usr_features


def compute_usr_similarity_between_groups(features_group1, models1, features_group2, models2):
    """Compute the USR similarity matrix between two model groups."""
    print("\n" + "=" * 80)
    print("[Computing between-group USR similarity matrix]")
    print("group1 models: {}, group2 models: {}".format(len(models1), len(models2)))
    print("=" * 80)

    all_feats = {}
    all_feats.update(features_group1)
    all_feats.update(features_group2)
    if not all_feats:
        raise RuntimeError("no USR features available for computation")

    max_len = max(v.shape[0] for v in all_feats.values())
    feature_dim = max_len * 27
    print("max residues L_max = {}, flattened dim = {}".format(max_len, feature_dim))

    def build_matrix(feat_dict, model_list):
        mat = np.zeros((len(model_list), feature_dim), dtype=np.float32)
        for i, name in enumerate(model_list):
            flat = feat_dict[name].reshape(-1)
            if flat.size > feature_dim:
                flat = flat[:feature_dim]
            elif flat.size < feature_dim:
                flat = np.pad(flat, (0, feature_dim - flat.size), mode='constant')
            mat[i] = flat
        return mat

    mat1 = build_matrix(features_group1, models1)
    mat2 = build_matrix(features_group2, models2)

    print("Computing distance matrix ({}, {}) ...".format(mat1.shape[0], mat2.shape[0]))
    distance_matrix = cdist(mat1, mat2, metric='euclidean')
    similarity_matrix = 1.0 / (1.0 + distance_matrix)

    print("Similarity matrix shape: {}".format(similarity_matrix.shape))
    print("Distance range: [{:.4f}, {:.4f}]".format(distance_matrix.min(), distance_matrix.max()))
    print("Similarity range: [{:.4f}, {:.4f}]".format(similarity_matrix.min(), similarity_matrix.max()))
    print("Mean similarity: {:.4f}".format(similarity_matrix.mean()))

    return similarity_matrix, distance_matrix


def write_mass_file(out_txt, target_name, top_k_names):
    """Write the selected models to a MASS-format prediction file."""
    
    with open(out_txt, "w", encoding="utf-8") as output:
        output.write("PFRMAT QA\n")
        output.write("TARGET " + target_name + "\n")
        output.write("METHOD QA \n")
        output.write("MODEL 2\n")
        output.write("QMODE 3\n")
        output.write(" ".join(top_k_names) + "\n")
        output.write("END")


def main():
    parser = argparse.ArgumentParser(
        description="USR-similarity-based consensus selection (score CSV to final MASS output)"
    )
    parser.add_argument("--target_name", required=True, help="target name")
    parser.add_argument("--all_model", required=True, help="model name list file")
    parser.add_argument("--G_csv", required=True, help="score CSV with (name, score)")
    parser.add_argument("--topk", type=int, default=5, help="number of anchor models (default 5)")
    parser.add_argument("--K", type=int, default=5, help="number of models to output (default 5)")
    parser.add_argument("--usr_feature_dir", required=True, help="USR feature directory (.npz files)")
    parser.add_argument("--output_dir", required=True, help="main output directory")
    parser.add_argument("--usr_out_dir", required=True, help="USR consensus result directory")
    parser.add_argument("--cpus", type=int, default=4, help="threads for feature loading")
    args = parser.parse_args()

    target_name = args.target_name
    all_model_file = args.all_model
    G_csv = args.G_csv
    topk = args.topk
    final_K = args.K
    usr_feature_dir = args.usr_feature_dir
    output_dir = args.output_dir
    usr_out_dir = args.usr_out_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(usr_out_dir, exist_ok=True)

    topk_file = os.path.join(output_dir, "topk.txt")
    fm_file = os.path.join(output_dir, "f_m.txt")
    sim_file = os.path.join(usr_out_dir, "usr_similarity_matrix.npy")
    mass_file = os.path.join(output_dir, "Select.mass.txt")

    # 1. Read all_model and G_csv to build the mapping.
    print("\n" + "=" * 80)
    print("[STEP1] Reading all_model file")
    if not os.path.exists(all_model_file):
        print("[ERROR] all_model file not found: {}".format(all_model_file))
        sys.exit(1)
    with open(all_model_file, 'r') as f:
        all_models_raw = [line.strip() for line in f.readlines() if line.strip()]
    print("  Total models: {}".format(len(all_models_raw)))

    norm_to_raw = {}
    for raw in all_models_raw:
        norm = normalize_massive_pred_name(raw)
        norm_to_raw[norm] = raw

    print("\n[STEP2] Reading score file {}".format(G_csv))
    if not os.path.exists(G_csv):
        print("[ERROR] score file not found: {}".format(G_csv))
        sys.exit(1)
    df = pd.read_csv(G_csv)

    # Detect column names.
    possible_name_cols = ['name', 'model', 'Name', 'Model', 'ID']
    possible_score_cols = ['score', 'Score']
    name_col = None
    score_col = None
    for c in df.columns:
        if c in possible_name_cols:
            name_col = c
            break
    for c in df.columns:
        if c in possible_score_cols:
            score_col = c
            break
    if name_col is None:
        name_col = df.columns[0]
    if score_col is None:
        if len(df.columns) >= 2:
            score_col = df.columns[1]
        else:
            print("[ERROR] unable to detect a score column")
            sys.exit(1)
    print("  name column: '{}', score column: '{}'".format(name_col, score_col))

    # Coerce scores to numeric and drop invalid rows.
    df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
    before = len(df)
    df = df.dropna(subset=[score_col])
    df = df[np.isfinite(df[score_col])]
    after = len(df)
    if before != after:
        print("[WARN] removed {} rows with invalid or non-numeric scores".format(before - after))

    norm_to_info = {}
    for idx, row in df.iterrows():
        g_name = str(row[name_col])
        score = row[score_col]
        norm = normalize_massive_pred_name(g_name)
        if norm not in norm_to_raw:
            continue
        raw = norm_to_raw[norm]
        norm_to_info[norm] = (raw, score)

    if not norm_to_info:
        print("[ERROR] no score entry matched any model; check naming rules")
        sys.exit(1)
    print("  Matched models: {}".format(len(norm_to_info)))

    # Sort by descending score.
    sorted_info = sorted(norm_to_info.items(), key=lambda x: x[1][1], reverse=True)

    # 2. Write topk.txt (small, regenerated each run).
    if os.path.exists(topk_file):
        print("\n[STEP4] topk.txt exists, reading")
        with open(topk_file, 'r') as f:
            topk_raw_names = [line.strip() for line in f.readlines() if line.strip()]
        print("  read {} models from file".format(len(topk_raw_names)))
    else:
        topk_entries = sorted_info[:topk]
        topk_raw_names = [info[0] for _, info in topk_entries]
        topk_scores = [info[1] for _, info in topk_entries]

        print("\n[STEP3] Top-{} high-score models (anchors):".format(topk))
        for i, name in enumerate(topk_raw_names):
            print("  {}. {} (score: {:.4f})".format(i+1, name, topk_scores[i]))

        with open(topk_file, 'w') as f:
            for name in topk_raw_names:
                f.write(name + "\n")
        print("Saved to {}".format(topk_file))

    # 3. Get the candidate list (f_m).
    if os.path.exists(fm_file):
        print("\n[STEP4] f_m.txt exists, reading")
        with open(fm_file, 'r') as f:
            fm_raw_names = [line.strip() for line in f.readlines() if line.strip()]
        print("  read {} models from file".format(len(fm_raw_names)))
    else:
        all_scores = [info[1] for _, info in sorted_info]
        median_score = np.median(all_scores)
        print("\n[STEP4] score median = {:.4f}".format(median_score))
        fm_entries = [(norm, info) for norm, info in sorted_info if info[1] > median_score]
        fm_raw_names = [info[0] for _, info in fm_entries]
        print("  models above median: {}".format(len(fm_raw_names)))
        with open(fm_file, 'w') as f:
            for name in fm_raw_names:
                f.write(name + "\n")
        print("Saved to {}".format(fm_file))

    if len(fm_raw_names) == 0:
        print("[ERROR] no model above the median, cannot continue")
        sys.exit(1)

    # 4. Get the similarity matrix (load if present).
    valid_models_csv = os.path.join(usr_out_dir, "valid_models.csv")
    topk_models_csv = os.path.join(usr_out_dir, "topk_models.csv")

    if os.path.exists(sim_file) and os.path.exists(valid_models_csv) and os.path.exists(topk_models_csv):
        print("\n[STEP5/6] similarity matrix and related files exist, loading")
        sim_matrix = np.load(sim_file)
        valid_models2_df = pd.read_csv(valid_models_csv)
        if 'model' in valid_models2_df.columns:
            valid_models2 = valid_models2_df['model'].tolist()
        else:
            valid_models2 = valid_models2_df.iloc[:, 0].tolist()
        topk_df = pd.read_csv(topk_models_csv)
        if 'model' in topk_df.columns:
            valid_models1 = topk_df['model'].tolist()
        else:
            valid_models1 = topk_df.iloc[:, 0].tolist()
        print("  matrix shape: {}, anchors: {}, candidates: {}".format(
            sim_matrix.shape, len(valid_models1), len(valid_models2)))
        if sim_matrix.shape != (len(valid_models1), len(valid_models2)):
            print("[ERROR] matrix shape disagrees with model lists; recomputing")
            sim_file_exists = False
        else:
            sim_file_exists = True
    else:
        sim_file_exists = False

    if not sim_file_exists:
        print("\n[STEP5/6] computing USR similarity matrix")
        models_set1 = topk_raw_names
        models_set2 = fm_raw_names
        unique_models = list(set(models_set1 + models_set2))
        print("loading USR features from {}".format(usr_feature_dir))
        all_usr_features = load_usr_features_single(usr_feature_dir, unique_models, num_workers=args.cpus)

        valid_models1 = [m for m in models_set1 if m in all_usr_features]
        valid_models2 = [m for m in models_set2 if m in all_usr_features]

        if not valid_models1:
            print("[ERROR] no valid anchor features, exiting")
            sys.exit(1)
        if not valid_models2:
            print("[ERROR] no valid candidate features, exiting")
            sys.exit(1)
        print("  valid anchors: {}, valid candidates: {}".format(len(valid_models1), len(valid_models2)))

        features1 = {m: all_usr_features[m] for m in valid_models1}
        features2 = {m: all_usr_features[m] for m in valid_models2}

        sim_matrix, dist_matrix = compute_usr_similarity_between_groups(
            features1, valid_models1, features2, valid_models2
        )

        np.save(sim_file, sim_matrix)
        print("Similarity matrix saved: {}".format(sim_file))
        pd.DataFrame({'model': valid_models2}).to_csv(valid_models_csv, index=False)
        pd.DataFrame({'model': valid_models1}).to_csv(topk_models_csv, index=False)
        print("Model name lists saved")

    # 5. Compute mean similarity (self excluded) and write the Select file.
    if os.path.exists(mass_file):
        print("\n[STEP7] Select.mass.txt exists, skipping generation.")
        print("=" * 80)
        print("USR similarity computation complete")
        print("=" * 80)
        return

    print("\n[STEP7] computing mean similarity (self excluded) and ranking")
    if sim_matrix.shape[0] == 0 or sim_matrix.shape[1] == 0:
        print("[ERROR] abnormal matrix shape, cannot compute mean")
        sys.exit(1)

    mean_similarities = []
    for j, decoy in enumerate(valid_models2):
        sims = sim_matrix[:, j]
        if decoy in valid_models1:
            anchor_idx = valid_models1.index(decoy)
            mask = np.ones(len(sims), dtype=bool)
            mask[anchor_idx] = False
            mean_val = sims[mask].mean()
        else:
            mean_val = sims.mean()
        mean_similarities.append(mean_val)

    model_scores = sorted(
        zip(valid_models2, mean_similarities),
        key=lambda x: x[1],
        reverse=True
    )

    select_csv = os.path.join(output_dir, "Select.mass.csv")
    df_select = pd.DataFrame(model_scores, columns=['model', 'mean_usr_similarity'])
    df_select.to_csv(select_csv, index=False)
    print("\nMean USR similarity ranking saved: {}".format(select_csv))

    top_k_selected = [name for name, _ in model_scores[:final_K]]
    if len(top_k_selected) < final_K:
        print("[WARN] fewer than {} valid models, outputting {} only".format(final_K, len(top_k_selected)))

    write_mass_file(mass_file, target_name, top_k_selected)
    print("MASS file saved: {}".format(mass_file))


if __name__ == "__main__":
    main()
