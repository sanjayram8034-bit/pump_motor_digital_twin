"""
Synthetic Data Generator for Industrial Centrifugal Pump.

Generates reproducible synthetic telemetry samples based on hydraulic/mechanical
operating principles and documented fault assumptions for predictive maintenance.

DISCLAIMER:
All generated data is strictly synthetic. Evaluation metrics derived from this
data must never be represented as real-world operational performance.
"""

from typing import Tuple
import numpy as np
import pandas as pd

# Explicit feature schema and ordering for Pump
PUMP_FEATURE_NAMES = [
    "temperature",
    "vibration",
    "current",
    "rpm",
    "pressure",
    "flow_rate"
]
PUMP_TARGET_NAME = "condition"
PUMP_CLASSES = ["Normal", "Warning", "Critical"]


def generate_pump_synthetic_data(
    n_samples_per_class: int = 1200,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic pump telemetry data with documented fault modes.

    Fault Assumptions:
    1. Normal:
       - Pump operates near the Best Efficiency Point (BEP).
       - Bearing housing temperature: 36 - 60 °C.
       - Vibration below ISO threshold (< 2.5 mm/s RMS).
       - Current: 9.5 - 16 A.
       - Discharge pressure: 3.8 - 5.5 Bar.
       - Flow rate: 70 - 96 m³/h.
    2. Warning (Hydraulic / Mechanical Degradation):
       - Mode A (Thermal-only): Bearing friction / seal warm-up (66 - 80 °C) with normal other readings.
       - Mode B (Vibration-only): Impeller imbalance / vane turbulence (2.8 - 6.5 mm/s RMS).
       - Mode C (Current-only): Minor mechanical binding / electrical rise (17 - 25 A).
       - Mode D (RPM-only): Motor speed slip (1390 - 1450 RPM).
       - Mode E (Pressure-only): Suction line restriction (1.8 - 3.2 Bar) or throttling backpressure (6.2 - 7.8 Bar).
       - Mode F (Flow-only): Degraded impeller delivery (32 - 62 m³/h) or excess runout (100 - 115 m³/h).
       - Mode G (Combined multi-fault): Coupled hydraulic and mechanical degradation.
    3. Critical (Cavitation, Dry Run, or Mechanical Seizure):
       - Mode A (Thermal-only): Severe bearing overheating / dry seal friction (86 - 125 °C).
       - Mode B (Vibration-only): Violent cavitation shockwaves / bearing seizure (7.2 - 18 mm/s RMS).
       - Mode C (Current-only): Severe motor overload (>28 A) OR fluid loss / no load (<4.5 A).
       - Mode D (RPM-only): Severe speed collapse / near stall (700 - 1360 RPM).
       - Mode E (Pressure-only): Loss of prime / suction loss (<0.8 Bar) OR dead-heading (>8.5 Bar).
       - Mode F (Flow-only): Fluid delivery collapse (<15 m³/h).
       - Mode G (Combined multi-fault): Severe suction cavitation, dry-running, or mechanical lockup.
    """
    rng = np.random.RandomState(random_seed)

    # Helper function to generate normal baseline arrays
    def _gen_normal_sub(n: int):
        t = np.clip(rng.normal(loc=48.0, scale=3.5, size=n), 36.0, 58.0)
        v = np.clip(rng.normal(loc=1.5, scale=0.3, size=n), 0.5, 2.3)
        c = np.clip(rng.normal(loc=13.0, scale=1.0, size=n), 9.5, 15.5)
        r = np.clip(rng.normal(loc=1475.0, scale=16.0, size=n), 1435.0, 1500.0)
        p = np.clip(rng.normal(loc=4.6, scale=0.3, size=n), 3.8, 5.4)
        q = np.clip(rng.normal(loc=84.0, scale=4.0, size=n), 72.0, 96.0)
        return t, v, c, r, p, q

    # 1. Normal Operation (n_samples_per_class)
    norm_t, norm_v, norm_c, norm_r, norm_p, norm_q = _gen_normal_sub(n_samples_per_class)
    df_normal = pd.DataFrame({
        "temperature": norm_t,
        "vibration": norm_v,
        "current": norm_c,
        "rpm": norm_r,
        "pressure": norm_p,
        "flow_rate": norm_q,
        "condition": "Normal"
    })

    # Divide fault classes into 7 distinct sub-scenarios
    sub_n = n_samples_per_class // 7
    rem_n = n_samples_per_class - (6 * sub_n)
    sub_counts = [sub_n] * 6 + [rem_n]

    # 2. Warning Operation
    # Sub 1: Thermal-only Warning (temp 66-80 °C, others normal)
    wt1 = np.clip(rng.normal(loc=72.0, scale=3.5, size=sub_counts[0]), 66.0, 80.0)
    _, wv1, wc1, wr1, wp1, wq1 = _gen_normal_sub(sub_counts[0])

    # Sub 2: Vibration-only Warning (vib 2.8-6.5 mm/s, others normal)
    wv2 = np.clip(rng.normal(loc=4.5, scale=0.75, size=sub_counts[1]), 2.8, 6.5)
    wt2, _, wc2, wr2, wp2, wq2 = _gen_normal_sub(sub_counts[1])

    # Sub 3: Electrical/Current-only Warning (curr 17-25 A, others normal)
    wc3 = np.clip(rng.normal(loc=20.0, scale=1.8, size=sub_counts[2]), 17.0, 25.0)
    wt3, wv3, _, wr3, wp3, wq3 = _gen_normal_sub(sub_counts[2])

    # Sub 4: RPM slip Warning (rpm 1250-1410 RPM, others normal)
    wr4 = np.clip(rng.normal(loc=1350.0, scale=25.0, size=sub_counts[3]), 1250.0, 1410.0)
    wt4, wv4, wc4, _, wp4, wq4 = _gen_normal_sub(sub_counts[3])

    # Sub 5: Pressure-only Warning (50% low pressure 1.8-3.2 Bar, 50% high backpressure 6.2-7.8 Bar)
    p_mask = rng.rand(sub_counts[4]) > 0.5
    wp5 = np.where(
        p_mask,
        np.clip(rng.normal(loc=2.5, scale=0.35, size=sub_counts[4]), 1.8, 3.2),
        np.clip(rng.normal(loc=6.9, scale=0.35, size=sub_counts[4]), 6.2, 7.8)
    )
    wt5, wv5, wc5, wr5, _, wq5 = _gen_normal_sub(sub_counts[4])

    # Sub 6: Flow-rate-only Warning (50% low delivery 35-62 m³/h, 50% runout 100-115 m³/h)
    q_mask = rng.rand(sub_counts[5]) > 0.5
    wq6 = np.where(
        q_mask,
        np.clip(rng.normal(loc=48.0, scale=5.0, size=sub_counts[5]), 32.0, 62.0),
        np.clip(rng.normal(loc=106.0, scale=4.0, size=sub_counts[5]), 100.0, 115.0)
    )
    wt6, wv6, wc6, wr6, wp6, _ = _gen_normal_sub(sub_counts[5])

    # Sub 7: Combined multi-fault Warning
    wt7 = np.clip(rng.normal(loc=71.0, scale=3.5, size=sub_counts[6]), 65.0, 80.0)
    wv7 = np.clip(rng.normal(loc=4.5, scale=0.75, size=sub_counts[6]), 2.8, 6.5)
    wc7 = np.clip(rng.normal(loc=19.5, scale=1.8, size=sub_counts[6]), 16.5, 25.0)
    wr7 = np.clip(rng.normal(loc=1350.0, scale=25.0, size=sub_counts[6]), 1250.0, 1410.0)
    wp7 = np.clip(rng.normal(loc=2.8, scale=0.4, size=sub_counts[6]), 1.8, 7.5)
    wq7 = np.clip(rng.normal(loc=54.0, scale=6.0, size=sub_counts[6]), 30.0, 115.0)

    df_warn = pd.DataFrame({
        "temperature": np.concatenate([wt1, wt2, wt3, wt4, wt5, wt6, wt7]),
        "vibration": np.concatenate([wv1, wv2, wv3, wv4, wv5, wv6, wv7]),
        "current": np.concatenate([wc1, wc2, wc3, wc4, wc5, wc6, wc7]),
        "rpm": np.concatenate([wr1, wr2, wr3, wr4, wr5, wr6, wr7]),
        "pressure": np.concatenate([wp1, wp2, wp3, wp4, wp5, wp6, wp7]),
        "flow_rate": np.concatenate([wq1, wq2, wq3, wq4, wq5, wq6, wq7]),
        "condition": "Warning"
    })

    # 3. Critical Operation
    # Sub 1: Thermal Critical (temp 86-125 °C, others normal)
    ct1 = np.clip(rng.normal(loc=98.0, scale=7.0, size=sub_counts[0]), 86.0, 125.0)
    _, cv1, cc1, cr1, cp1, cq1 = _gen_normal_sub(sub_counts[0])

    # Sub 2: Vibration Shock Critical (vib 7.2-18 mm/s, others normal)
    cv2 = np.clip(rng.normal(loc=10.2, scale=2.0, size=sub_counts[1]), 7.2, 18.0)
    ct2, _, cc2, cr2, cp2, cq2 = _gen_normal_sub(sub_counts[1])

    # Sub 3: Extreme Current Critical (50% locked >28A, 50% fluid loss <4.5A)
    c_crit_mask = rng.rand(sub_counts[2]) > 0.5
    cc3 = np.where(
        c_crit_mask,
        np.clip(rng.normal(loc=32.0, scale=3.0, size=sub_counts[2]), 28.0, 45.0),
        np.clip(rng.normal(loc=3.2, scale=0.6, size=sub_counts[2]), 1.5, 4.5)
    )
    ct3, cv3, _, cr3, cp3, cq3 = _gen_normal_sub(sub_counts[2])

    # Sub 4: Severe Speed Stall Critical (rpm 600-1200 RPM, others normal)
    cr4 = np.clip(rng.normal(loc=950.0, scale=80.0, size=sub_counts[3]), 600.0, 1200.0)
    ct4, cv4, cc4, _, cp4, cq4 = _gen_normal_sub(sub_counts[3])

    # Sub 5: Extreme Pressure Critical (50% loss <0.8 Bar, 50% deadhead >8.5 Bar)
    p_crit_mask = rng.rand(sub_counts[4]) > 0.5
    cp5 = np.where(
        p_crit_mask,
        np.clip(rng.normal(loc=0.4, scale=0.15, size=sub_counts[4]), 0.0, 0.8),
        np.clip(rng.normal(loc=9.5, scale=0.5, size=sub_counts[4]), 8.5, 12.0)
    )
    ct5, cv5, cc5, cr5, _, cq5 = _gen_normal_sub(sub_counts[4])

    # Sub 6: Flow Collapse Critical (flow <15 m³/h)
    cq6 = np.clip(rng.normal(loc=6.0, scale=2.5, size=sub_counts[5]), 0.0, 14.5)
    ct6, cv6, cc6, cr6, cp6, _ = _gen_normal_sub(sub_counts[5])

    # Sub 7: Combined Cavitation Dry-Run / Seizure Critical
    ct7 = np.clip(rng.normal(loc=95.0, scale=7.0, size=sub_counts[6]), 86.0, 125.0)
    cv7 = np.clip(rng.normal(loc=10.0, scale=2.0, size=sub_counts[6]), 7.2, 18.0)
    cc7 = np.clip(rng.normal(loc=3.8, scale=0.8, size=sub_counts[6]), 1.5, 45.0)
    cr7 = np.clip(rng.normal(loc=1250.0, scale=90.0, size=sub_counts[6]), 700.0, 1380.0)
    cp7 = np.clip(rng.normal(loc=0.4, scale=0.2, size=sub_counts[6]), 0.0, 12.0)
    cq7 = np.clip(rng.normal(loc=5.0, scale=2.5, size=sub_counts[6]), 0.0, 15.0)

    df_crit = pd.DataFrame({
        "temperature": np.concatenate([ct1, ct2, ct3, ct4, ct5, ct6, ct7]),
        "vibration": np.concatenate([cv1, cv2, cv3, cv4, cv5, cv6, cv7]),
        "current": np.concatenate([cc1, cc2, cc3, cc4, cc5, cc6, cc7]),
        "rpm": np.concatenate([cr1, cr2, cr3, cr4, cr5, cr6, cr7]),
        "pressure": np.concatenate([cp1, cp2, cp3, cp4, cp5, cp6, cp7]),
        "flow_rate": np.concatenate([cq1, cq2, cq3, cq4, cq5, cq6, cq7]),
        "condition": "Critical"
    })

    # Combine and shuffle
    df_combined = pd.concat([df_normal, df_warn, df_crit], ignore_index=True)
    df_shuffled = df_combined.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    return df_shuffled


def get_pump_features_and_target(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract features in strict canonical order and target column."""
    X = df[PUMP_FEATURE_NAMES].copy()
    y = df[PUMP_TARGET_NAME].copy()
    return X, y


if __name__ == "__main__":
    df = generate_pump_synthetic_data()
    print(f"Generated {len(df)} synthetic pump records.")
    print("Class distribution:")
    print(df["condition"].value_counts())
    print("\nFeature summary:")
    print(df.groupby("condition").mean())
