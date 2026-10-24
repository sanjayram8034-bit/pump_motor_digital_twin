"""
Synthetic Data Generator for Industrial Induction Motor.

Generates reproducible synthetic telemetry samples based on physical operating
principles and documented fault assumptions for predictive maintenance.

DISCLAIMER:
All generated data is strictly synthetic. Evaluation metrics derived from this
data must never be represented as real-world operational performance.
"""

from typing import Tuple
import numpy as np
import pandas as pd

# Explicit feature schema and ordering for Motor
MOTOR_FEATURE_NAMES = ["temperature", "vibration", "current", "rpm"]
MOTOR_TARGET_NAME = "condition"
MOTOR_CLASSES = ["Normal", "Warning", "Critical"]


def generate_motor_synthetic_data(
    n_samples_per_class: int = 1200,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic motor telemetry data with documented fault modes.

    Fault Assumptions:
    1. Normal:
       - Motor operates near rated load and sync speed (1450-1490 RPM).
       - Bearing vibration within ISO 10816 Zone A/B (< 2.5 mm/s RMS).
       - Temperature rise within standard thermal rating (38 - 68 °C).
       - Balanced phase current within nominal rated limits (9.5 - 17 A).
    2. Warning (Incipient Degraded States):
       - Mode A (Thermal-only): High cooling fin fouling / ambient heat (72 - 88 °C) with normal vib/current/speed.
       - Mode B (Vibration-only): Early bearing outer raceway defect (2.8 - 6.5 mm/s RMS) with normal temp/current/speed.
       - Mode C (Electrical-only): Phase imbalance or overload current (17.5 - 26 A) with normal temp/vib/speed.
       - Mode D (Speed-only): Excessive mechanical slip (1390 - 1450 RPM) with normal temp/vib/current.
       - Mode E (Multi-fault): Classic combined degradation across all parameters.
    3. Critical (Severe Failure Modes):
       - Mode A (Thermal runaway): Critical insulation limit exceeded (90 - 135 °C) with normal/moderate other readings.
       - Mode B (Vibration shock): Severe bearing seizure / fluting in ISO Zone D (7.2 - 18 mm/s RMS).
       - Mode C (Severe overcurrent): Near locked-rotor / phase short threshold (27 - 48 A).
       - Mode D (Severe stall): Severe RPM collapse / near-stall slip (700 - 1380 RPM).
       - Mode E (Combined severe): Multi-parameter catastrophic failure.
    """
    rng = np.random.RandomState(random_seed)

    # Helper function to generate normal baseline arrays
    def _gen_normal_sub(n: int):
        t = np.clip(rng.normal(loc=55.0, scale=4.0, size=n), 38.0, 68.0)
        v = np.clip(rng.normal(loc=1.6, scale=0.35, size=n), 0.5, 2.5)
        c = np.clip(rng.normal(loc=13.8, scale=1.1, size=n), 9.5, 17.0)
        r = np.clip(rng.normal(loc=1475.0, scale=16.0, size=n), 1435.0, 1500.0)
        return t, v, c, r

    # 1. Normal Operation (n_samples_per_class)
    norm_temp, norm_vib, norm_curr, norm_rpm = _gen_normal_sub(n_samples_per_class)
    df_normal = pd.DataFrame({
        "temperature": norm_temp,
        "vibration": norm_vib,
        "current": norm_curr,
        "rpm": norm_rpm,
        "condition": "Normal"
    })

    # Divide fault classes into 5 distinct sub-scenarios
    sub_n = n_samples_per_class // 5
    rem_n = n_samples_per_class - (4 * sub_n)
    sub_counts = [sub_n, sub_n, sub_n, sub_n, rem_n]

    # 2. Warning Operation
    # Sub 1: Thermal-only Warning (temp 72-88 °C, others normal)
    t1 = np.clip(rng.normal(loc=78.0, scale=3.5, size=sub_counts[0]), 72.0, 88.0)
    _, v1, c1, r1 = _gen_normal_sub(sub_counts[0])

    # Sub 2: Vibration-only Warning (vib 2.8-6.5 mm/s, others normal)
    v2 = np.clip(rng.normal(loc=4.6, scale=0.7, size=sub_counts[1]), 2.8, 6.5)
    t2, _, c2, r2 = _gen_normal_sub(sub_counts[1])

    # Sub 3: Electrical/Current-only Warning (curr 18-26 A, others normal)
    c3 = np.clip(rng.normal(loc=21.5, scale=1.5, size=sub_counts[2]), 18.0, 26.0)
    t3, v3, _, r3 = _gen_normal_sub(sub_counts[2])

    # Sub 4: RPM / Speed slip Warning (rpm 1250-1410 RPM, others normal)
    r4 = np.clip(rng.normal(loc=1350.0, scale=25.0, size=sub_counts[3]), 1250.0, 1410.0)
    t4, v4, c4, _ = _gen_normal_sub(sub_counts[3])

    # Sub 5: Combined multi-fault Warning
    t5 = np.clip(rng.normal(loc=78.0, scale=3.5, size=sub_counts[4]), 70.0, 88.0)
    v5 = np.clip(rng.normal(loc=4.6, scale=0.7, size=sub_counts[4]), 2.8, 6.5)
    c5 = np.clip(rng.normal(loc=21.5, scale=1.5, size=sub_counts[4]), 17.5, 26.0)
    r5 = np.clip(rng.normal(loc=1350.0, scale=25.0, size=sub_counts[4]), 1250.0, 1410.0)

    df_warn = pd.DataFrame({
        "temperature": np.concatenate([t1, t2, t3, t4, t5]),
        "vibration": np.concatenate([v1, v2, v3, v4, v5]),
        "current": np.concatenate([c1, c2, c3, c4, c5]),
        "rpm": np.concatenate([r1, r2, r3, r4, r5]),
        "condition": "Warning"
    })

    # 3. Critical Operation
    # Sub 1: Thermal Runaway Critical (temp 92-135 °C, others normal)
    ct1 = np.clip(rng.normal(loc=102.0, scale=6.0, size=sub_counts[0]), 91.0, 135.0)
    _, cv1, cc1, cr1 = _gen_normal_sub(sub_counts[0])

    # Sub 2: Severe Vibration Shock Critical (vib 7.2-18 mm/s, others normal)
    cv2 = np.clip(rng.normal(loc=10.0, scale=1.8, size=sub_counts[1]), 7.2, 18.0)
    ct2, _, cc2, cr2 = _gen_normal_sub(sub_counts[1])

    # Sub 3: Severe Overcurrent Critical (curr 28-48 A, others normal)
    cc3 = np.clip(rng.normal(loc=34.0, scale=3.5, size=sub_counts[2]), 28.0, 48.0)
    ct3, cv3, _, cr3 = _gen_normal_sub(sub_counts[2])

    # Sub 4: Severe Stall / RPM Collapse Critical (rpm 600-1200 RPM, others normal)
    cr4 = np.clip(rng.normal(loc=950.0, scale=80.0, size=sub_counts[3]), 600.0, 1200.0)
    ct4, cv4, cc4, _ = _gen_normal_sub(sub_counts[3])

    # Sub 5: Combined Catastrophic Critical
    ct5 = np.clip(rng.normal(loc=102.0, scale=6.0, size=sub_counts[4]), 90.0, 135.0)
    cv5 = np.clip(rng.normal(loc=10.0, scale=1.8, size=sub_counts[4]), 7.2, 18.0)
    cc5 = np.clip(rng.normal(loc=34.0, scale=3.5, size=sub_counts[4]), 28.0, 48.0)
    cr5 = np.clip(rng.normal(loc=950.0, scale=80.0, size=sub_counts[4]), 600.0, 1200.0)

    df_crit = pd.DataFrame({
        "temperature": np.concatenate([ct1, ct2, ct3, ct4, ct5]),
        "vibration": np.concatenate([cv1, cv2, cv3, cv4, cv5]),
        "current": np.concatenate([cc1, cc2, cc3, cc4, cc5]),
        "rpm": np.concatenate([cr1, cr2, cr3, cr4, cr5]),
        "condition": "Critical"
    })

    # Combine and shuffle
    df_combined = pd.concat([df_normal, df_warn, df_crit], ignore_index=True)
    df_shuffled = df_combined.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    return df_shuffled


def get_motor_features_and_target(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract features in strict canonical order and target column."""
    X = df[MOTOR_FEATURE_NAMES].copy()
    y = df[MOTOR_TARGET_NAME].copy()
    return X, y


if __name__ == "__main__":
    df = generate_motor_synthetic_data()
    print(f"Generated {len(df)} synthetic motor records.")
    print("Class distribution:")
    print(df["condition"].value_counts())
    print("\nFeature summary:")
    print(df.groupby("condition").mean())
