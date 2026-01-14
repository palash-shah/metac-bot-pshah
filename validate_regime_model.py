#!/usr/bin/env python3
"""
Validate regime_model.pkl is correct and ready to use.
"""

import pickle
import numpy as np


def main() -> bool:
    print("=" * 80)
    print("REGIME MODEL VALIDATION")
    print("=" * 80)

    # [1] Load the model
    print("\n[1/4] Loading regime_model.pkl...")
    try:
        with open("regime_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("❌ regime_model.pkl not found")
        print("   Run: python build_regime_params.py")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

    # [2] Validate transition matrix
    print("\n[2/4] Validating transition matrix...")
    transition_matrix = model_data.get("transition_matrix")
    if not transition_matrix:
        print("❌ No transition_matrix in model_data")
        return False

    for rid, row in transition_matrix.items():
        total = sum(row.values())
        print(f"  Regime {rid}: {row}")
        if abs(total - 1.0) > 1e-3:
            print(f"  ❌ Probabilities do not sum to 1.0 (sum={total})")
            return False
        else:
            print(f"  ✅ Valid row (sum={total:.3f})")

    # [3] Validate regime anchors
    print("\n[3/4] Validating regime anchors...")
    anchors = model_data.get("regime_anchors")
    if not anchors:
        print("❌ No regime_anchors in model_data")
        return False

    for rid, a in anchors.items():
        print(f"\n  Regime {rid}:")
        print(f"    binary_default: {a.get('binary_default')}")
        print(f"    sample_size: {a.get('sample_size')}")
        bd = a.get("binary_default", 0.5)
        if not (0.0 <= bd <= 1.0):
            print("    ❌ binary_default out of [0,1]")
            return False
        print("    ✅ Anchors look OK")

    # [4] Inspect spectral model fit (no .predict here)
    print("\n[4/4] Checking spectral clustering labels...")
    spectral = model_data.get("spectral_model")
    if spectral is None:
        print("❌ No spectral_model in model_data")
        return False

    try:
        labels = getattr(spectral, "labels_", None)
        if labels is None:
            print("❌ spectral_model.labels_ is missing (model may not be fitted)")
            return False
        labels = np.array(labels)
        print(f"  labels_ shape: {labels.shape}")
        print(f"  Unique regimes: {np.unique(labels)}")
    except Exception as e:
        print(f"❌ Error inspecting spectral_model.labels_: {e}")
        return False

    print("\n" + "=" * 80)
    print("✅ ALL VALIDATION CHECKS PASSED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
