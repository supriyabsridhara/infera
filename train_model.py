"""
Infera – Component 2 & 3: Bayesian Network Training + Inference
Constructs a Discrete Bayesian Network, learns CPTs from data,
then saves the model so the dashboard and evaluator can reload it.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteMLE
from pgmpy.inference import VariableElimination

RISK_THRESHOLDS = {"low": 0.30, "high": 0.70}


def load_data(path: str = os.path.join("data", "sensor_data.csv")) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Keep only the four model variables (integer-encoded)
    return df[["entry_status", "door_sensor", "motion_sensor", "glass_sensor"]]


def build_and_train(df: pd.DataFrame) -> DiscreteBayesianNetwork:
    """
    Network structure:
        entry_status → door_sensor
        entry_status → motion_sensor
        entry_status → glass_sensor
    """
    model = DiscreteBayesianNetwork(
        [
            ("entry_status", "door_sensor"),
            ("entry_status", "motion_sensor"),
            ("entry_status", "glass_sensor"),
        ]
    )
    # In pgmpy >=1.1, fit() returns the fitted model
    fitted = model.fit(df, estimator=DiscreteMLE())
    assert fitted.check_model(), "Model structure/CPT validation failed!"
    return fitted


def infer(model: DiscreteBayesianNetwork, door: int, motion: int, glass: int) -> dict:
    """
    Returns a dict:
        {class: probability, ..., risk_level: str, recommended_action: str}
    """
    ve = VariableElimination(model)
    evidence = {
        "door_sensor":   door,
        "motion_sensor": motion,
        "glass_sensor":  glass,
    }
    result = ve.query(variables=["entry_status"], evidence=evidence, show_progress=False)
    states = result.state_names["entry_status"]
    values = result.values

    probs = {state: float(val) for state, val in zip(states, values)}

    intrusion_p = probs.get("intrusion", 0.0)

    if intrusion_p >= RISK_THRESHOLDS["high"]:
        risk = "High"
        action = "Immediately notify security personnel and law enforcement."
    elif intrusion_p >= RISK_THRESHOLDS["low"]:
        risk = "Moderate"
        action = "Dispatch security guard to verify the situation."
    else:
        risk = "Low"
        action = "Continue normal monitoring. No immediate action required."

    return {**probs, "risk_level": risk, "recommended_action": action}


def save_model(model: DiscreteBayesianNetwork, path: str = os.path.join("data", "bayesian_model.pkl")):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved  : {path}")


def load_model(path: str = os.path.join("data", "bayesian_model.pkl")) -> DiscreteBayesianNetwork:
    with open(path, "rb") as f:
        return pickle.load(f)


def print_cpds(model: DiscreteBayesianNetwork):
    print("\n  Conditional Probability Tables (CPTs):")
    print("  " + "-" * 50)
    for cpd in model.get_cpds():
        print()
        print(f"  Node: {cpd.variable}")
        print(cpd)


def main():
    os.makedirs("data", exist_ok=True)

    print("=" * 55)
    print("  Infera – Model Training")
    print("=" * 55)

    df = load_data()
    print(f"  Loaded {len(df):,} records for training.")

    model = build_and_train(df)
    print("  Bayesian Network trained successfully.")
    save_model(model)

    print_cpds(model)

    # Quick sanity inference
    print("\n  Sample inference – door=1, motion=1, glass=1:")
    result = infer(model, door=1, motion=1, glass=1)
    for cls in ["no_entry", "authorized", "intrusion"]:
        print(f"    P({cls:<11}) = {result[cls]:.4f}")
    print(f"    Risk level : {result['risk_level']}")
    print(f"    Action     : {result['recommended_action']}")


if __name__ == "__main__":
    main()
