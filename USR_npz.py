#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import locale
import warnings

try:
    os.environ['LC_ALL'] = 'C'
    locale.setlocale(locale.LC_ALL, 'C')
except locale.Error as e:
    print(f"[Locale Warning] {e}")

warnings.filterwarnings("ignore", message=".*A NumPy version.*")
import numpy as np
import torch
from Bio import PDB
from scipy.spatial.distance import cdist

def extract_coordinates_from_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    n_coords, ca_coords, c_coords, centroids = [], [], [], []

    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    try:
                        n = residue['N'].coord
                        ca = residue['CA'].coord
                        c = residue['C'].coord
                        atoms = [atom for atom in residue.get_atoms() if atom.element != 'H']
                        coords = np.array([atom.coord for atom in atoms])
                        centroid = np.mean(coords, axis=0)
                        n_coords.append(n)
                        ca_coords.append(ca)
                        c_coords.append(c)
                        centroids.append(centroid)
                    except KeyError as e:
                        print(f"Missing atom {e}, skip residue {residue.get_resname()} {residue.id} chain {chain.id}")
                        print(f"File: {pdb_file}")
                        continue

    L = len(ca_coords)
    maps = torch.zeros((1, L, 4, 3))
    maps[0, :, 0, :] = torch.tensor(n_coords)
    maps[0, :, 1, :] = torch.tensor(ca_coords)
    maps[0, :, 2, :] = torch.tensor(c_coords)
    return maps, np.array(ca_coords), np.array(centroids)

def compute_cb_coordinates(maps):
    N = maps[:, :, 0, :]
    Ca = maps[:, :, 1, :]
    C = maps[:, :, 2, :]
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca
    return Cb  # [1, L, 3]

def extract_USR(distance: torch.Tensor) -> torch.Tensor:
    distance = torch.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)
    B, L, _ = distance.shape

    avg1 = torch.mean(distance, dim=2).clamp(min=1e-6)
    idx2 = torch.argmax(distance, dim=2)
    q2_rows = distance[torch.arange(B)[:, None], idx2, :]
    avg2 = torch.mean(q2_rows, dim=2).clamp(min=1e-6)
    idx3 = torch.argmax(q2_rows, dim=2)
    q3_rows = distance[torch.arange(B)[:, None], idx3, :]
    avg3 = torch.mean(q3_rows, dim=2).clamp(min=1e-6)

    var1 = torch.var(distance, dim=2).clamp(min=1e-6)
    var2 = torch.var(q2_rows, dim=2).clamp(min=1e-6)
    var3 = torch.var(q3_rows, dim=2).clamp(min=1e-6)

    std1 = torch.std(distance, dim=2).clamp(min=1e-6)
    std2 = torch.std(q2_rows, dim=2).clamp(min=1e-6)
    std3 = torch.std(q3_rows, dim=2).clamp(min=1e-6)

    usr = torch.stack([avg1, avg2, avg3, var1, var2, var3, std1, std2, std3], dim=2)
    return usr  # [B, L, 9]

def process_one_pdb(pdb_path, save_path):
    maps, ca_coords, centroids = extract_coordinates_from_pdb(pdb_path)

    cb_coords = compute_cb_coordinates(maps).squeeze(0).numpy()

    ca_ca_dist = cdist(ca_coords, ca_coords)
    cb_cb_dist = cdist(cb_coords, cb_coords)
    centroid_dist = cdist(centroids, centroids)

    ca_tensor = torch.tensor(ca_ca_dist, dtype=torch.float32).unsqueeze(0)
    cb_tensor = torch.tensor(cb_cb_dist, dtype=torch.float32).unsqueeze(0)
    centroid_tensor = torch.tensor(centroid_dist, dtype=torch.float32).unsqueeze(0)

    usr_ca = extract_USR(ca_tensor)
    usr_cb = extract_USR(cb_tensor)
    usr_centroid = extract_USR(centroid_tensor)

    usr_combined = torch.cat([usr_ca, usr_cb, usr_centroid], dim=2)  # [1, L, 27]

    np.savez(save_path, ca_ca=ca_ca_dist, cb_cb=cb_cb_dist, centroid=centroid_dist, usr=usr_combined.numpy())

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir', type=str, default=None,
                        help="kept for compatibility; PDB paths come from --model_list_path")
    parser.add_argument('--save_dir', type=str, required=True,
                        help="USR feature root; writes to save_dir/<target_name>/")
    parser.add_argument('--model_list_path', type=str, required=True,
                        help="file with one full PDB path per line (multiple dirs allowed)")
    parser.add_argument('--target_name', type=str, required=True,
                        help="target name; all features go to save_dir/<target_name>/")
    parser.add_argument('--max_files', type=int, default=None, help="process only the first N files (debug)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.model_list_path, 'r') as f:
        pdb_list = [line.strip() for line in f if line.strip()]

    count = 0
    processed_count = 0
    skipped_count = 0
    error_count = 0

    print(f"\nTotal {len(pdb_list)} files to process")
    print(f"Output dir: {args.save_dir}\n")

    for pdb_path in pdb_list:
        original_path = pdb_path.strip()
        pdb_path = original_path

        # File path may be given with or without .pdb.
        if os.path.isfile(pdb_path):
            pass
        elif os.path.isfile(pdb_path + ".pdb"):
            pdb_path = pdb_path + ".pdb"
        elif pdb_path.endswith(".pdb") and os.path.isfile(pdb_path[:-4]):
            pdb_path = pdb_path[:-4]
        else:
            print(f"File not found: {original_path} (tried: {original_path}, {original_path}.pdb)")
            skipped_count += 1
            continue

        if args.max_files is not None and count >= args.max_files:
            print(f"\nReached debug limit {args.max_files}, stopping")
            break
        count += 1

        target_dir = os.path.join(args.save_dir, args.target_name)
        os.makedirs(target_dir, exist_ok=True)

        # npz stem = PDB basename without .pdb, matching downstream name + '.npz'.
        pdb_basename = os.path.basename(pdb_path)
        stem = pdb_basename[:-4] if pdb_basename.lower().endswith('.pdb') else pdb_basename
        save_path = os.path.join(target_dir, stem + '.npz')

        print(f"\n[{count}/{len(pdb_list)}] Processing: {pdb_path}")
        print(f"   TARGET: {args.target_name}")
        print(f"   Save to: {save_path}")

        try:
            process_one_pdb(pdb_path, save_path)
            processed_count += 1
        except Exception as e:
            print(f"Failed: {pdb_path}")
            print(f"   Error: {str(e)}")
            error_count += 1
            import traceback
            traceback.print_exc()

    print("\n" + "="*50)
    print("Summary:")
    print(f"   processed : {processed_count}")
    print(f"   skipped   : {skipped_count}")
    print(f"   errors    : {error_count}")
    print(f"   output dir: {args.save_dir}")
    print("="*50)
