"""
Detect landmark pair swaps between processed_r1_data and processed_ave_data.

A swap is flagged when, for a given VL_ID, landmark A in r1 has prone-transformed
coordinates closer to landmark B in ave than to landmark A in ave (and vice versa).

Outputs a table of all swapped (vl_id, lm_name_r1, matched_ave_name) rows.
"""

import pandas as pd
import numpy as np
from pathlib import Path

EXCEL_PATH = Path("../output/landmark_results_v8_2026_03_16.xlsx")

PRONE_COLS = ["landmark prone transformed x",
              "landmark prone transformed y",
              "landmark prone transformed z"]


def load_sheet(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()
    return df


def find_swaps(r1_df, ave_df, label="r1"):
    """
    For each (VL_ID, landmark name) in r1, find the closest landmark in ave
    (same VL_ID) by prone-transformed coords. Flag as swap if the closest
    ave landmark has a different name.
    """
    swaps = []

    for vl_id, r1_group in r1_df.groupby("VL_ID"):
        ave_group = ave_df[ave_df["VL_ID"] == vl_id]
        if ave_group.empty:
            continue

        # Drop rows with missing coords
        r1_coords = r1_group[PRONE_COLS].values
        ave_coords = ave_group[PRONE_COLS].values

        r1_names = r1_group["Landmark name"].values
        ave_names = ave_group["Landmark name"].values

        # Skip if any NaN
        if np.any(np.isnan(r1_coords)) or np.any(np.isnan(ave_coords)):
            continue

        for j, (r1_name, r1_coord) in enumerate(zip(r1_names, r1_coords)):
            dists = np.linalg.norm(ave_coords - r1_coord, axis=1)
            closest_idx = np.argmin(dists)
            closest_ave_name = ave_names[closest_idx]
            closest_dist = dists[closest_idx]

            # Self-distance: distance to same name in ave
            same_name_mask = ave_names == r1_name
            if same_name_mask.any():
                same_dist = np.linalg.norm(
                    ave_coords[same_name_mask][0] - r1_coord)
            else:
                same_dist = np.inf

            is_swap = (closest_ave_name != r1_name) and (closest_dist < same_dist - 1.0)

            if is_swap:
                swaps.append({
                    "VL_ID": vl_id,
                    f"{label}_name": r1_name,
                    "closest_ave_name": closest_ave_name,
                    f"dist_to_closest_ave ({closest_ave_name})": round(closest_dist, 2),
                    f"dist_to_same_ave ({r1_name})": round(same_dist, 2),
                })

    return pd.DataFrame(swaps)


def main():
    print(f"Reading {EXCEL_PATH} ...")
    r1_df = load_sheet(EXCEL_PATH, "processed_r1_data")
    r2_df = load_sheet(EXCEL_PATH, "processed_r2_data")
    ave_df = load_sheet(EXCEL_PATH, "processed_ave_data")

    print(f"  r1 rows: {len(r1_df)}, r2 rows: {len(r2_df)}, ave rows: {len(ave_df)}")

    print("\n=== Swaps in r1 vs ave ===")
    swaps_r1 = find_swaps(r1_df, ave_df, label="r1")
    if swaps_r1.empty:
        print("  None detected.")
    else:
        print(swaps_r1.to_string(index=False))

    print("\n=== Swaps in r2 vs ave ===")
    swaps_r2 = find_swaps(r2_df, ave_df, label="r2")
    if swaps_r2.empty:
        print("  None detected.")
    else:
        print(swaps_r2.to_string(index=False))

    # Summary
    print("\n=== Summary ===")
    print(f"r1 swapped rows: {len(swaps_r1)}")
    print(f"r2 swapped rows: {len(swaps_r2)}")

    if not swaps_r1.empty:
        print("\nAffected VL_IDs (r1):", sorted(swaps_r1["VL_ID"].unique().tolist()))

    # Check r1 == r2 swaps (same pairs?)
    if not swaps_r1.empty and not swaps_r2.empty:
        r1_pairs = set(zip(swaps_r1["VL_ID"], swaps_r1["r1_name"]))
        r2_pairs = set(zip(swaps_r2["VL_ID"], swaps_r2["r2_name"]))
        shared = r1_pairs & r2_pairs
        print(f"\nSwaps shared between r1 and r2: {len(shared)} pairs")


if __name__ == "__main__":
    main()
