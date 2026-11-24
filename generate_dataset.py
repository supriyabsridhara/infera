"""
Infera – Component 1: Sensor Data Generator
Generates a 20,000-record simulated building security dataset.
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_RECORDS = 20_000

np.random.seed(RANDOM_SEED)

# ── Class prior probabilities ────────────────────────────────────────────────
# 50 % no-entry, 35 % authorized, 15 % intrusion
CLASS_PROBS = {"no_entry": 0.50, "authorized": 0.35, "intrusion": 0.15}

# ── Conditional probabilities P(sensor=1 | entry_status) ────────────────────
# Format: {entry_status: (P_door, P_motion, P_glass)}
COND_PROBS = {
    "no_entry":   (0.03, 0.05, 0.01),   # very low – occasional false positives
    "authorized": (0.90, 0.85, 0.04),   # door/motion almost always, glass rarely
    "intrusion":  (0.80, 0.90, 0.75),   # door/motion/glass all likely
}


def generate_dataset(n: int = N_RECORDS) -> pd.DataFrame:
    classes = list(CLASS_PROBS.keys())
    probs   = list(CLASS_PROBS.values())

    entry_status = np.random.choice(classes, size=n, p=probs)

    door_arr   = np.zeros(n, dtype=int)
    motion_arr = np.zeros(n, dtype=int)
    glass_arr  = np.zeros(n, dtype=int)

    for i, status in enumerate(entry_status):
        p_door, p_motion, p_glass = COND_PROBS[status]
        door_arr[i]   = int(np.random.rand() < p_door)
        motion_arr[i] = int(np.random.rand() < p_motion)
        glass_arr[i]  = int(np.random.rand() < p_glass)

    df = pd.DataFrame({
        "entry_status": entry_status,
        "door_sensor":  door_arr,
        "motion_sensor":motion_arr,
        "glass_sensor": glass_arr,
    })

    # Readable binary labels for display
    df["door_sensor_label"]   = df["door_sensor"].map({0: "No", 1: "Yes"})
    df["motion_sensor_label"] = df["motion_sensor"].map({0: "No", 1: "Yes"})
    df["glass_sensor_label"]  = df["glass_sensor"].map({0: "No", 1: "Yes"})

    return df


def main():
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    path = os.path.join("data", "sensor_data.csv")
    df.to_csv(path, index=False)

    print("=" * 55)
    print("  Infera – Dataset Generation Complete")
    print("=" * 55)
    print(f"  Records      : {len(df):,}")
    print(f"  Saved to     : {path}")
    print()
    print("  Class distribution:")
    for cls, cnt in df["entry_status"].value_counts().items():
        pct = cnt / len(df) * 100
        print(f"    {cls:<14} {cnt:>6,}  ({pct:.1f} %)")
    print()
    print("  Sample rows:")
    print(df[["entry_status", "door_sensor", "motion_sensor", "glass_sensor"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
