#!/usr/bin/env python3
"""
Train the regime model (spectral clustering) on training questions.

This script:
1. Fetches 30-40 training questions from TRAIN_URLS
2. Extracts features (forecast, volatility, age, activity, crowd size)
3. Trains spectral clustering (n_clusters=2)
4. Computes regime anchors
5. Saves regime_model.pkl for use in main.py

Run once to create regime_model.pkl, then run main.py to use it.
"""

import os
import json
import pickle
from datetime import datetime
from statistics import mean, pstdev

import dotenv
import numpy as np
from sklearn.cluster import SpectralClustering

from forecasting_tools import (
    MetaculusClient,
    BinaryQuestion,
    NumericQuestion,
    MultipleChoiceQuestion,
    MetaculusQuestion,
)

dotenv.load_dotenv()

# Initialize client (reads METACULUS_TOKEN from env)
client = MetaculusClient()

# ============================================================================
# TRAINING QUESTIONS
# ============================================================================

TRAIN_URLS = [
    "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
    "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
    "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
    "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
    "https://www.metaculus.com/questions/600/will-a-sample-of-negative-energy-be-produced-by-2100/",
    "https://www.metaculus.com/questions/6304/us-semiconductor-fab-capacity-jan-2030/",
    "https://www.metaculus.com/questions/1493/global-population-decline-10-by-2100/",
    "https://www.metaculus.com/questions/6365/riemann-h-proved-true-if-settled-by-2100/",
    "https://www.metaculus.com/questions/1432/sustainable-off-earth-human-colony-by-2100/",
    "https://www.metaculus.com/questions/1646/human-condition-to-change-before-2100/",
    "https://www.metaculus.com/questions/24630/qubit-count-by-2050/",
    "https://www.metaculus.com/questions/40776/will-any-ai-achieve-94-on-gpqa-diamond-benchmark-leaderboard-by-feb-1-2026/",
    "https://www.metaculus.com/questions/13896/no-of-asat-tests-by-2030/",
    "https://www.metaculus.com/questions/39330/what-will-be-the-percentage-increase-for-the-minimum-wage-in-colombia-for-2026/",
    "https://www.metaculus.com/questions/21188/ai-chip-smuggling-countries/",
    "https://www.metaculus.com/questions/38770/will-the-iranian-government-lose-power-before-2027/",
    "https://www.metaculus.com/questions/41339/best-traditional-country-album-at-2026-grammys/",
    "https://www.metaculus.com/questions/41495/annual-number-of-active-usa-protests-from-2026-to-2029/",
    "https://www.metaculus.com/questions/28211/number-of-teams-in-2050/",
    "https://www.metaculus.com/questions/20362/xprize-healthspan-grand-prize-award/",
    "https://www.metaculus.com/questions/22056/highest-gpqa-diamond-score-2024-to-2027/",
    "https://www.metaculus.com/questions/14542/number-of-endangered-languages-at-the-end-of-2030/",
    "https://www.metaculus.com/questions/40237/when-will-the-golden-dome-missile-defense-system-be-operational/",
    "https://www.metaculus.com/questions/18612/us-military-intervention-in-mexico-by-2029/",
    "https://www.metaculus.com/questions/40416/gop-senate-advantage-after-2026-midterms/",
    "https://www.metaculus.com/questions/16553/ai-blackmail-for-material-gain-by-eoy-2028/",
    "https://www.metaculus.com/questions/38666/what-will-the-yoy-rent-increase-be-in-nyc-in-december-2026-according-to-zillow/",
    "https://www.metaculus.com/questions/8393/new-start-renewed-until-february-2027/",
    "https://www.metaculus.com/questions/40872/will-there-be-a-successful-coup-in-africa-or-latin-america-by-march-1-2026/",
    "https://www.metaculus.com/questions/40413/gop-house-advantage-after-2026-midterms/",
    "https://www.metaculus.com/questions/41205/will-openai-file-an-s-1-before-march-15-2026/",
    "https://www.metaculus.com/questions/40742/when-will-oxfordshire-uk-huge-plastic-fly-tip-be-cleared/",
    "https://www.metaculus.com/questions/41502/will-the-us-gain-sovereignty-over-any-part-of-greenland-in-2026/",
    "https://www.metaculus.com/questions/40884/trump-declare-war-on-iran-before-2029/",
    "https://www.metaculus.com/questions/19879/will-tesla-mass-produce-humanoid-robots/",
    "https://www.metaculus.com/questions/10614/elon-musk-becomes-worlds-first-trillionaire/",
    "https://www.metaculus.com/questions/38783/will-donald-trump-attempt-to-deport-elon-musk-before-july-1-2026/",
]

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(question: dict) -> np.ndarray:
    """
    Extract numerical features from a question for spectral clustering.
    """
    community_values = question.get("community_prediction", [])
    if community_values and len(community_values) > 0:
        latest = community_values[-1]
        community_forecast = latest / 100.0 if isinstance(latest, (int, float)) else 0.5
    else:
        community_forecast = 0.5
    community_forecast = max(0.0, min(1.0, community_forecast))

    if community_values and len(community_values) > 3:
        recent = np.array(community_values[-10:]) / 100.0
        volatility = float(np.std(recent))
        volatility = min(1.0, volatility)
    else:
        volatility = 0.1

    created_str = question.get("created", "")
    if created_str:
        try:
            created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            now = datetime.now(created.tzinfo) if created.tzinfo else datetime.now()
            age_days = (now - created).days
            age = min(1.0, max(0.0, age_days / 365.0))
        except Exception as e:
            print(f"[WARNING] Error parsing date {created_str}: {e}")
            age = 0.5
    else:
        age = 0.5

    if community_values and len(community_values) > 0:
        activity = min(1.0, max(0.0, len(community_values) / 100.0))
    else:
        activity = 0.1

    activity_data = question.get("activity_count", 0)
    crowd_size = min(1.0, max(0.0, activity_data / 50.0))

    features = np.array(
        [community_forecast, volatility, age, activity, crowd_size],
        dtype=np.float32,
    )
    return features

# ============================================================================
# REGIME ANCHOR COMPUTATION
# ============================================================================

def compute_regime_anchors(questions: list, regime_labels: np.ndarray) -> dict:
    regimes = {}
    n_regimes = len(np.unique(regime_labels))

    for regime_id in range(n_regimes):
        mask = regime_labels == regime_id
        regime_questions = [q for i, q in enumerate(questions) if mask[i]]

        if len(regime_questions) == 0:
            print(f"[WARNING] Regime {regime_id} has no questions, skipping")
            continue

        forecasts = []
        for q in regime_questions:
            community_values = q.get("community_prediction", [])
            if community_values and len(community_values) > 0:
                latest = community_values[-1]
                forecast = latest / 100.0 if isinstance(latest, (int, float)) else 0.5
                forecasts.append(max(0.0, min(1.0, forecast)))
            else:
                forecasts.append(0.5)

        forecasts_array = np.array(forecasts)
        regimes[regime_id] = {
            "binary_default": float(np.median(forecasts_array)),
            "numeric_scale": 1.0,
            "numeric_shift": 0.0,
            "multiple_choice_confidence": 0.35,
            "sample_size": len(regime_questions),
            "mean_forecast": float(np.mean(forecasts_array)),
            "std_forecast": float(np.std(forecasts_array)) if len(forecasts_array) > 1 else 0.0,
        }

    return regimes

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("REGIME MODEL TRAINING")
    print("=" * 80)

    print("\n[1/4] Fetching training questions...")
    questions = []
    for url in TRAIN_URLS:
        try:
            q: MetaculusQuestion = client.get_question_by_url(url)
            if q is None:
                print(f"  ✗ No data for {url}")
                continue

            community_series: list[float] = []

            # 1) For binary questions, use point CP at access time if present (0–1)
            if isinstance(q, BinaryQuestion) and q.community_prediction_at_access_time is not None:
                community_series = [q.community_prediction_at_access_time * 100.0]

            # 2) Optionally, use any stored history in custom_metadata
            try:
                if not community_series and hasattr(q, "custom_metadata"):
                    cp_hist = q.custom_metadata.get("community_prediction_history", [])
                    if isinstance(cp_hist, list):
                        community_series = cp_hist
            except Exception:
                pass

            question_dict = {
                "title": getattr(q, "question_text", ""),
                "community_prediction": community_series,
                "created": (
                    q.published_time.isoformat() if getattr(q, "published_time", None) else ""
                ),
                "activity_count": getattr(q, "num_predictions", None) or 0,
            }

            questions.append(question_dict)
            print(f"  ✓ {question_dict['title'][:60]}")

        except Exception as e:
            print(f"  ✗ Failed to fetch {url}: {e}")

    print(f"\nLoaded {len(questions)} training questions")

    if len(questions) < 10:
        print(f"ERROR: Need at least 10 questions to train. Got {len(questions)}")
        print("Please add more URLs to TRAIN_URLS")
        return False

    print("\n[2/4] Extracting features from questions...")
    X = np.array([extract_features(q) for q in questions])
    print(f"Feature matrix shape: {X.shape}")
    print("Feature matrix sample (first 3 rows):")
    for i in range(min(3, len(X))):
        print(f"  Question {i}: {X[i]}")

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("ERROR: Feature matrix contains NaN or Inf values")
        print(f"NaN count: {np.sum(np.isnan(X))}")
        print(f"Inf count: {np.sum(np.isinf(X))}")
        return False

    print("\n[3/4] Training spectral clustering model...")
    try:
        spectral = SpectralClustering(
            n_clusters=2,
            affinity="rbf",
            random_state=42,
            n_init=10,
        )
        regime_labels = spectral.fit_predict(X)

        print(f"Regime assignments: {regime_labels}")
        print(f"Regime 0 count: {np.sum(regime_labels == 0)}")
        print(f"Regime 1 count: {np.sum(regime_labels == 1)}")
    except Exception as e:
        print(f"ERROR training spectral clustering: {e}")
        return False

    # ---- NEW: compute cluster centers for runtime prediction ----
    print("\n[3b/4] Computing cluster centers in feature space...")
    try:
        n_clusters = 2
        cluster_centers = []
        for k in range(n_clusters):
            mask = regime_labels == k
            if np.any(mask):
                center = X[mask].mean(axis=0)
            else:
                center = X.mean(axis=0)
            cluster_centers.append(center)
        cluster_centers = np.vstack(cluster_centers)
        print(f"Cluster centers shape: {cluster_centers.shape}")
    except Exception as e:
        print(f"ERROR computing cluster centers: {e}")
        return False

    print("\n[4/4] Computing regime anchors...")
    try:
        regime_anchors = compute_regime_anchors(questions, regime_labels)
        for regime_id, anchors in regime_anchors.items():
            print(f"\nRegime {regime_id}:")
            print(f"  binary_default: {anchors['binary_default']:.3f}")
            print(f"  mean_forecast: {anchors['mean_forecast']:.3f}")
            print(f"  std_forecast: {anchors['std_forecast']:.3f}")
            print(f"  sample_size: {anchors['sample_size']}")
    except Exception as e:
        print(f"ERROR computing regime anchors: {e}")
        return False

    transition_matrix = {
        0: {0: 0.9, 1: 0.1},
        1: {0: 0.1, 1: 0.9},
    }

    print("\n[SAVE] Saving regime model to regime_model.pkl...")
    try:
        model_data = {
            "spectral_model": spectral,
            "regime_anchors": regime_anchors,
            "transition_matrix": transition_matrix,
            "n_regimes": 2,
            "feature_names": ["community_forecast", "volatility", "age", "activity", "crowd_size"],
            "training_questions_count": len(questions),
            "training_timestamp": datetime.now().isoformat(),
            "cluster_centers": cluster_centers,
        }

        with open("regime_model.pkl", "wb") as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("✓ Saved regime_model.pkl")
    except Exception as e:
        print(f"ERROR saving regime_model.pkl: {e}")
        return False

    print("[SAVE] Saving summary to regime_model_summary.json...")
    try:
        summary = {
            "n_regimes": 2,
            "transition_matrix": transition_matrix,
            "regime_anchors": {
                str(k): {
                    "binary_default": float(v["binary_default"]),
                    "numeric_scale": float(v["numeric_scale"]),
                    "numeric_shift": float(v["numeric_shift"]),
                    "multiple_choice_confidence": float(v["multiple_choice_confidence"]),
                    "sample_size": v["sample_size"],
                    "mean_forecast": v["mean_forecast"],
                    "std_forecast": v["std_forecast"],
                }
                for k, v in regime_anchors.items()
            },
            "training_questions": len(questions),
            "training_timestamp": datetime.now().isoformat(),
            "feature_names": ["community_forecast", "volatility", "age", "activity", "crowd_size"],
        }

        with open("regime_model_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("✓ Saved regime_model_summary.json")
    except Exception as e:
        print(f"ERROR saving regime_model_summary.json: {e}")
        return False

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Verify files: ls -lh regime_model.pkl regime_model_summary.json")
    print("2. Validate model: python validate_regime_model.py")
    print("3. Test bot: python main.py --mode test_questions")
    print("4. Deploy: python main.py --mode tournament")

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
