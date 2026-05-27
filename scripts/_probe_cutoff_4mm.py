"""
Throwaway probe: compute which VL subjects/landmark pairs become valid
in find_corresponding_landmarks if the 3 mm cutoff is relaxed to 4 mm.

Reads output/observer_landmarks_comparison.xlsx (sheets 'anthony', 'holly')
which is also the input to landmark_curation_analysis.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

XL = Path(__file__).resolve().parent.parent / "output" / "observer_landmarks_comparison.xlsx"

a = pd.read_excel(XL, sheet_name="anthony")
h = pd.read_excel(XL, sheet_name="holly")
print(f"Anthony rows: {len(a)}   Holly rows: {len(h)}")

# Merge on Subject + Filename (same scheme as landmark_curation_analysis.py)
m = pd.merge(
    a, h, on=["Subject", "Filename"], how="inner",
    suffixes=(" (anthony)", " (holly)"),
)
print(f"Merged inner rows: {len(m)}")

# Drop rejected by either
m = m[
    (m.get("Status (anthony)") != "rejected")
    & (m.get("Status (holly)") != "rejected")
].copy()

lt_a = m["Landmark Type (anthony)"]
lt_h = m["Landmark Type (holly)"]

# Drop fibroadenoma + require type match
keep = (lt_a != "fibroadenoma") & (lt_h != "fibroadenoma") & (lt_a == lt_h)
m = m[keep].copy()
print(f"After curation (no rejected, no fibroadenoma, type match): {len(m)}")

dxp = m["Prone X (holly)"] - m["Prone X (anthony)"]
dyp = m["Prone Y (holly)"] - m["Prone Y (anthony)"]
dzp = m["Prone Z (holly)"] - m["Prone Z (anthony)"]
m["prone_dist"] = np.sqrt(dxp**2 + dyp**2 + dzp**2)

dxs = m["Supine X (holly)"] - m["Supine X (anthony)"]
dys = m["Supine Y (holly)"] - m["Supine Y (anthony)"]
dzs = m["Supine Z (holly)"] - m["Supine Z (anthony)"]
m["supine_dist"] = np.sqrt(dxs**2 + dys**2 + dzs**2)

print()
print(
    f"Prone dist:  max = {m['prone_dist'].max():.4f} mm   "
    f"(>0.001 mm: {(m['prone_dist'] > 0.001).sum()} rows)"
)
print(
    f"Supine dist: median = {m['supine_dist'].median():.2f} mm   "
    f"max = {m['supine_dist'].max():.2f} mm"
)


def subjects_passing(cutoff: float):
    ok = m[(m["prone_dist"] <= cutoff) & (m["supine_dist"] <= cutoff)]
    return set(ok["Subject"].unique()), ok


s3, ok3 = subjects_passing(3.0)
s4, ok4 = subjects_passing(4.0)

print()
print(f"Subjects with >=1 valid pair at 3 mm cutoff:  {len(s3)}")
print(f"Subjects with >=1 valid pair at 4 mm cutoff:  {len(s4)}")
added = sorted(s4 - s3)
print(f"NEW subjects added at 4 mm (relative to 3 mm): {len(added)}")
for s in added:
    rows = ok4[ok4["Subject"] == s]
    n_new_band = ((rows["supine_dist"] > 3.0) | (rows["prone_dist"] > 3.0)).sum()
    print(
        f"  {s}: {len(rows)} valid pair(s) at 4mm   "
        f"(of which {n_new_band} are in the 3<d<=4 band)"
    )

print()
print("Subjects already at 3mm — extra pairs gained going 3 -> 4:")
gains = []
for s in sorted(s3):
    n3 = len(ok3[ok3["Subject"] == s])
    n4 = len(ok4[ok4["Subject"] == s])
    if n4 > n3:
        gains.append((s, n3, n4, n4 - n3))
for s, n3, n4, d in gains:
    print(f"  {s}: {n3} -> {n4} pairs  (+{d})")
print(f"Total extra pairs in already-included subjects: {sum(d for _,_,_,d in gains)}")
print(f"Total pairs at 3mm: {len(ok3)}   |   at 4mm: {len(ok4)}   (+{len(ok4)-len(ok3)})")
