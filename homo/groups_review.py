#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, re, sys, tempfile, random
import numpy as np, pandas as pd
from os.path import exists, join

RANDOM_SEED = 20260226  # default grouping seed; override with --seed

def atomic_write_text(path, content):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=d, prefix=".tmp.", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(content)
        tmp = f.name
    os.replace(tmp, path)

def atomic_write_dataframe(path, df):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=d, prefix=".tmp.", suffix=".csv", delete=False, encoding="utf-8") as f:
        df.to_csv(f, index=False)
        tmp = f.name
    os.replace(tmp, path)

# Name normalization for score matching.
def normalize_model_name(filename):
    name = os.path.basename(str(filename).strip()).replace(".pdb", "")
    name = name.replace(".pdb", "")
    name = re.sub(r'^ranked_\d+_', '', name)
    name = re.sub(r'_ranked_\d+_unrelaxed', '', name)
    return name

def csv2dict(in_csv, name_col, score_col, clean2orig):
    df = pd.read_csv(in_csv)
    if name_col not in df.columns or score_col not in df.columns:
        raise ValueError(f"{in_csv} missing columns {name_col}/{score_col}")
    model_dict = {}
    for raw_name, score in zip(df[name_col], df[score_col]):
        try:
            s = float(score)
        except (ValueError, TypeError):
            continue
        c = normalize_model_name(raw_name)
        if c in clean2orig:
            model_dict[clean2orig[c]] = s
    return model_dict

# Rank a group by each metric.
def rank_by_metrics(model_list, g_dict, mv_dict):
    g_sort = sorted(model_list, key=lambda x: g_dict.get(x, 0.0), reverse=True)
    mv_sort = sorted(model_list, key=lambda x: mv_dict.get(x, 0.0), reverse=True)
    print("=== Top-1 model per metric within group ===")
    if g_sort:
        print(f"G top1: {g_sort[0]} | score={g_dict.get(g_sort[0], 0.0):.6f}")
    if mv_sort:
        print(f"MV top1: {mv_sort[0]} | score={mv_dict.get(mv_sort[0], 0.0):.6f}")
    return [g_sort, mv_sort]

# Consensus scoring (self-similarity excluded).
def con_order(order_list, sim_mat, name2idx, score_dict):
    for idx_i, mi in enumerate(order_list):
        if mi not in name2idx:
            continue
        i = name2idx[mi]
        scores = []
        for idx_j, mj in enumerate(order_list):
            if idx_i == idx_j or mj not in name2idx:
                continue
            j = name2idx[mj]
            scores.append(float(sim_mat[i, j]))
        s = float(np.mean(scores)) if scores else 0.0
        if mi not in score_dict or s > score_dict[mi]:
            score_dict[mi] = s
    return score_dict

def load_similarity_matrix(npy_path, valid_csv, model_list):
    full = np.load(npy_path)
    dfv = pd.read_csv(valid_csv)
    valid_names = dfv['model_name'].tolist()
    name2idx = {n: i for i, n in enumerate(valid_names)}
    indices, missing = [], []
    for m in model_list:
        if m in name2idx:
            indices.append(name2idx[m])
        else:
            missing.append(m)
    if missing:
        print(f"Warning: {len(missing)} models not found in full similarity matrix")
        model_list = [m for m in model_list if m in name2idx]
    if not indices:
        raise ValueError("no model found in full similarity matrix")
    sub = full[indices][:, indices]
    new_map = {m: i for i, m in enumerate(model_list)}
    return sub, new_map

# Output.
def build_final_output(score_dict, csv_path, txt_path, target):
    items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(items, columns=["raw_name", "consensus_usr_score"])
    atomic_write_dataframe(csv_path, df)
    print(f"[OUTPUT] CSV written: {csv_path}")
    top5 = df["raw_name"].head(5).tolist()
    content = (
        "PFRMAT QA\n"
        f"TARGET {target}\n"
        "METHOD QA \n"
        "MODEL 2\nQMODE 3\n"
        f"{' '.join(top5)}\nEND"
    )
    atomic_write_text(txt_path, content)

def split_groups(lst, n, seed):
    random.seed(seed)
    shuffled = lst.copy()
    random.shuffle(shuffled)
    groups = {}
    total = len(shuffled)
    base = total // n
    rem = total % n
    idx = 0
    for gid in range(n):
        sz = base + 1 if gid < rem else base
        groups[gid] = shuffled[idx:idx+sz]
        idx += sz
    return groups

def main():
    parser = argparse.ArgumentParser(description="G+MV dual-metric grouped consensus (with name normalization)")
    parser.add_argument("--stepn_csv", required=True, help="candidate pool CSV, column: name")
    parser.add_argument("--g_csv", required=True, help="G score table, columns: name, score")
    parser.add_argument("--mv_csv", required=True, help="MV score table, columns: name, score")
    parser.add_argument("--valid_names_csv", required=True, help="valid model name list")
    parser.add_argument("--group_num", type=int, default=8, help="number of groups")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help=f"grouping seed (default {RANDOM_SEED})")
    parser.add_argument("--usr_sm", required=True, help="USR similarity matrix .npy")
    parser.add_argument("--target_name", required=True, help="target name")
    parser.add_argument("--out_dir", default=".", help="output directory")
    args = parser.parse_args()

    for p in [args.stepn_csv, args.g_csv, args.mv_csv, args.valid_names_csv, args.usr_sm]:
        if not exists(p):
            print(f"[ERROR] file not found: {p}")
            sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n[STEP1] Reading candidate pool")
    df_s = pd.read_csv(args.stepn_csv)
    if 'name' not in df_s.columns:
        print(f"[ERROR] {args.stepn_csv} missing 'name' column")
        sys.exit(1)
    raw_models = df_s['name'].dropna().tolist()
    clean2orig = {}
    for raw in raw_models:
        c = normalize_model_name(raw)
        clean2orig[c] = raw
    print(f"Candidates: {len(raw_models)}, unique after normalization: {len(clean2orig)}")

    print("\n[STEP2] Loading G/MV score tables")
    g_dict = csv2dict(args.g_csv, "name", "score", clean2orig)
    mv_dict = csv2dict(args.mv_csv, "name", "score", clean2orig)
    print(f"G matched models: {len(g_dict)}")
    print(f"MV matched models: {len(mv_dict)}")

    print("\n[STEP3] Loading similarity matrix")
    sim_mat, name2idx = load_similarity_matrix(args.usr_sm, args.valid_names_csv, raw_models)
    print(f"Matrix shape: {sim_mat.shape}")

    print(f"\n[STEP4] Splitting into {args.group_num} groups")
    groups = split_groups(raw_models, args.group_num, args.seed)
    info = []
    for gid, mlist in groups.items():
        info.append({"group_id": gid, "model_count": len(mlist), "model_list": ",".join(mlist)})
    atomic_write_dataframe(join(args.out_dir, "group_info.csv"), pd.DataFrame(info))

    print("\n[STEP5] Within-group consensus")
    new_order = {}
    max_len = 0
    for gid, gmodels in groups.items():
        if not gmodels: continue
        [g_list, mv_list] = rank_by_metrics(gmodels, g_dict, mv_dict)
        order = []
        scores = {}
        cur_max = max(len(g_list), len(mv_list))
        max_len = max(max_len, cur_max)
        for th in range(cur_max):
            if th < len(g_list):
                order.append(g_list[th])
            if th < len(mv_list):
                order.append(mv_list[th])
            scores = con_order(order, sim_mat, name2idx, scores)
        if scores:
            sorted_list = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            new_order[gid] = sorted_list
            print(f"  group {gid}: kept {len(sorted_list)} models")

    if not new_order:
        print("[SKIP] no valid within-group consensus result")
        sys.exit(1)

    print("\n[STEP6] Cross-group global consensus")
    final_order = []
    final_scores = {}
    for th in range(max_len):
        for _, flist in new_order.items():
            if th < len(flist):
                final_order.append(flist[th])
        final_scores = con_order(final_order, sim_mat, name2idx, final_scores)

    out_csv = join(args.out_dir, "Select.mass.csv")
    out_txt = join(args.out_dir, "Select.mass.txt")
    build_final_output(final_scores, out_csv, out_txt, args.target_name)

if __name__ == "__main__":
    main()
