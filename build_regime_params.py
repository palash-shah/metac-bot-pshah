#!/usr/bin/env python3
"""
Train the regime model (spectral clustering) on training questions.

This script:
1. Fetches 30-40 training questions from TRAIN_URLS
2. Extracts 7 structural features (time horizon, age, crowd size, confidence, complexity, type, activity)
3. Trains spectral clustering (n_clusters=3)
4. Computes regime anchors
5. Saves regime_model.pkl for use in main.py

Run once to create regime_model.pkl, then run main.py to use it.
"""

import os
import json
import pickle
from datetime import datetime, timezone
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
# FEATURE EXTRACTION - NEW 7-FEATURE MODEL
# ============================================================================

def extract_features(question: dict, question_obj: MetaculusQuestion) -> np.ndarray:
    """
    Extract 7 structural features from a question for spectral clustering.

    Features:
    1. Time horizon (0-1): days until resolution / 730 days
    2. Question age (0-1): days since published / 365 days
    3. Crowd size (0-1): log10(num_predictions) / 3.0
    4. Community confidence (0-1): distance from 0.5 * 2
    5. Complexity (0-1): (word_count - 50) / 150
    6. Question type (0/0.5/1): binary/numeric/multiple_choice
    7. Activity rate (0-1): predictions_per_day / 5.0
    """

    # Feature 1: Time horizon (days until resolution)
    time_horizon = 0.5  # default for unknown
    if hasattr(question_obj, 'resolve_time') and question_obj.resolve_time:
        try:
            now = datetime.now(timezone.utc)
            days_to_resolution = (question_obj.resolve_time - now).days
            # Normalize: 0 days = 0, 730 days (2 years) = 1.0
            time_horizon = min(1.0, max(0.0, days_to_resolution / 730.0))
        except Exception as e:
            print(f"[WARNING] Error computing time_horizon: {e}")
            time_horizon = 0.5

    # Feature 2: Question age (days since published)
    age = 0.5
    created_str = question.get("created", "")
    if created_str:
        try:
            created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            now = datetime.now(created.tzinfo) if created.tzinfo else datetime.now()
            age_days = (now - created).days
            # Normalize: 0 days = 0, 365 days = 1.0
            age = min(1.0, max(0.0, age_days / 365.0))
        except Exception as e:
            print(f"[WARNING] Error parsing date {created_str}: {e}")
            age = 0.5

    # Feature 3: Crowd size (log scale)
    n_preds = question.get("activity_count", 0)
    if n_preds > 0:
        # Log scale: 1 predictor = 0, 10 = 0.33, 100 = 0.67, 1000 = 1.0
        crowd_size = min(1.0, np.log10(n_preds) / 3.0)
    else:
        crowd_size = 0.0

    # Feature 4: Community confidence (distance from 0.5)
    community_confidence = 0.0
    community_values = question.get("community_prediction", [])
    if community_values and len(community_values) > 0:
        latest = community_values[-1]
        if isinstance(latest, (int, float)):
            p = latest / 100.0  # Convert to 0-1
            # Distance from 0.5, scaled to 0-1
            community_confidence = abs(p - 0.5) * 2.0

    # Feature 5: Question complexity (word count proxy)
    title = question.get("title", "")
    resolution_criteria = question.get("resolution_criteria", "")
    full_text = title + " " + resolution_criteria
    word_count = len(full_text.split())
    # Normalize: 50 words = 0 (simple), 200+ words = 1.0 (complex)
    complexity = min(1.0, max(0.0, (word_count - 50) / 150))

    # Feature 6: Question type encoding
    qtype = question.get("type", "binary")
    if qtype == "binary":
        type_encoding = 0.0
    elif qtype == "numeric":
        type_encoding = 0.5
    elif qtype == "multiple_choice":
        type_encoding = 1.0
    else:
        type_encoding = 0.25  # Unknown type

    # Feature 7: Activity rate (predictions per day)
    age_days = max(1, age * 365)  # Avoid division by zero
    activity_rate = min(1.0, n_preds / age_days / 5.0)  # 5 predictions/day = 1.0

    return np.array([
        time_horizon,
        age,
        crowd_size,
        community_confidence,
        complexity,
        type_encoding,
        activity_rate
    ], dtype=np.float32)

# ============================================================================
# REGIME ANCHOR COMPUTATION - ENHANCED
# ============================================================================

def compute_regime_anchors(questions: list, question_objects: list, regime_labels: np.ndarray) -> dict:
    """
    Compute regime-specific priors based on question characteristics.
    """
    regimes = {}
    n_regimes = len(np.unique(regime_labels))

    for regime_id in range(n_regimes):
        mask = regime_labels == regime_id
        regime_questions = [q for i, q in enumerate(questions) if mask[i]]
        regime_objs = [q for i, q in enumerate(question_objects) if mask[i]]

        if len(regime_questions) == 0:
            print(f"[WARNING] Regime {regime_id} has no questions, skipping")
            continue

        # Compute binary forecasts for this regime
        binary_forecasts = []
        for q in regime_questions:
            cp = q.get("community_prediction", [])
            if cp and len(cp) > 0:
                latest = cp[-1]
                if isinstance(latest, (int, float)):
                    binary_forecasts.append(latest / 100.0)

        # Compute regime characteristics
        time_horizons = []
        crowd_sizes = []
        complexities = []

        for q_obj, q_dict in zip(regime_objs, regime_questions):
            # Time horizon
            if hasattr(q_obj, 'resolve_time') and q_obj.resolve_time:
                try:
                    days = (q_obj.resolve_time - datetime.now(timezone.utc)).days
                    time_horizons.append(max(0, days))
                except:
                    pass

            # Crowd size
            crowd_sizes.append(q_dict.get("activity_count", 0))

            # Complexity
            title = q_dict.get("title", "")
            complexities.append(len(title.split()))

        # Compute regime anchors
        binary_forecasts_arr = np.array(binary_forecasts) if binary_forecasts else np.array([0.5])

        regimes[regime_id] = {
            # Binary question anchors
            "binary_default": float(np.median(binary_forecasts_arr)),
            "binary_mean": float(np.mean(binary_forecasts_arr)),
            "binary_std": float(np.std(binary_forecasts_arr)) if len(binary_forecasts_arr) > 1 else 0.2,

            # Numeric question adjustments (scale and shift)
            "numeric_scale": 1.0 + float(np.std(binary_forecasts_arr)) if len(binary_forecasts_arr) > 1 else 1.0,
            "numeric_shift": 0.0,

            # Multiple choice confidence
            "multiple_choice_confidence": 0.35,

            # Regime characteristics (for analysis)
            "sample_size": len(regime_questions),
            "avg_time_horizon_days": float(np.mean(time_horizons)) if time_horizons else 365,
            "median_time_horizon_days": float(np.median(time_horizons)) if time_horizons else 365,
            "avg_crowd_size": float(np.mean(crowd_sizes)) if crowd_sizes else 10,
            "avg_complexity": float(np.mean(complexities)) if complexities else 50,

            # Adaptive weight based on sample size
            "confidence_weight": 0.3 + 0.2 * (len(regime_questions) / len(questions)),
        }

    return regimes

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("REGIME MODEL TRAINING - 7-FEATURE MODEL WITH 3 REGIMES")
    print("=" * 80)

    print("\n[1/4] Fetching training questions...")
    questions = []
    question_objects = []

    for url in TRAIN_URLS:
        try:
            q: MetaculusQuestion = client.get_question_by_url(url)
            if q is None:
                print(f"  ✗ No data for {url}")
                continue

            # Store the question object
            question_objects.append(q)

            # Extract community prediction data
            community_series: list[float] = []
            if isinstance(q, BinaryQuestion) and q.community_prediction_at_access_time is not None:
                community_series = [q.community_prediction_at_access_time * 100.0]

            # Try to get historical data if available
            try:
                if not community_series and hasattr(q, "custom_metadata"):
                    cp_hist = q.custom_metadata.get("community_prediction_history", [])
                    if isinstance(cp_hist, list):
                        community_series = cp_hist
            except Exception:
                pass

            # Determine question type
            if isinstance(q, BinaryQuestion):
                qtype = "binary"
            elif isinstance(q, NumericQuestion):
                qtype = "numeric"
            elif isinstance(q, MultipleChoiceQuestion):
                qtype = "multiple_choice"
            else:
                qtype = "other"

            question_dict = {
                "title": getattr(q, "question_text", ""),
                "community_prediction": community_series,
                "created": q.published_time.isoformat() if getattr(q, "published_time", None) else "",
                "activity_count": getattr(q, "num_predictions", None) or 0,
                "type": qtype,
                "resolution_criteria": getattr(q, "resolution_criteria", ""),
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

    print("\n[2/4] Extracting 7 features from questions...")
    X = np.array([
        extract_features(q_dict, q_obj)
        for q_dict, q_obj in zip(questions, question_objects)
    ])
    print(f"Feature matrix shape: {X.shape}")
    print("Feature names: time_horizon, age, crowd_size, community_confidence, complexity, type, activity_rate")
    print("\nFeature matrix sample (first 3 rows):")
    for i in range(min(3, len(X))):
        print(f"  Question {i}: {X[i]}")

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("ERROR: Feature matrix contains NaN or Inf values")
        print(f"NaN count: {np.sum(np.isnan(X))}")
        print(f"Inf count: {np.sum(np.isinf(X))}")
        return False

    print("\n[3/4] Training spectral clustering model (3 regimes)...")
    try:
        spectral = SpectralClustering(
            n_clusters=3,  # CHANGED to 3 regimes
            affinity="rbf",
            random_state=42,
            n_init=10,
        )
        regime_labels = spectral.fit_predict(X)

        print(f"Regime assignments: {regime_labels}")
        print(f"Regime 0 count: {np.sum(regime_labels == 0)}")
        print(f"Regime 1 count: {np.sum(regime_labels == 1)}")
        print(f"Regime 2 count: {np.sum(regime_labels == 2)}")
    except Exception as e:
        print(f"ERROR training spectral clustering: {e}")
        return False

    # Compute cluster centers for runtime prediction
    print("\n[3b/4] Computing cluster centers in feature space...")
    try:
        n_clusters = 3
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
        print("Cluster centers:")
        for k in range(n_clusters):
            print(f"  Regime {k}: {cluster_centers[k]}")
    except Exception as e:
        print(f"ERROR computing cluster centers: {e}")
        return False

    print("\n[4/4] Computing regime anchors...")
    try:
        regime_anchors = compute_regime_anchors(questions, question_objects, regime_labels)
        for regime_id, anchors in regime_anchors.items():
            print(f"\n{'='*60}")
            print(f"Regime {regime_id} Characteristics:")
            print(f"{'='*60}")
            print(f"  Sample size: {anchors['sample_size']}")
            print(f"  Avg time horizon: {anchors['avg_time_horizon_days']:.0f} days")
            print(f"  Median time horizon: {anchors['median_time_horizon_days']:.0f} days")
            print(f"  Avg crowd size: {anchors['avg_crowd_size']:.0f} predictors")
            print(f"  Avg complexity: {anchors['avg_complexity']:.0f} words")
            print(f"  Binary default: {anchors['binary_default']:.3f}")
            print(f"  Binary mean: {anchors['binary_mean']:.3f}")
            print(f"  Binary std: {anchors['binary_std']:.3f}")
            print(f"  Confidence weight: {anchors['confidence_weight']:.3f}")
    except Exception as e:
        print(f"ERROR computing regime anchors: {e}")
        return False

    # Create transition matrix for 3 regimes
    transition_matrix = {
        0: {0: 0.85, 1: 0.10, 2: 0.05},
        1: {0: 0.10, 1: 0.85, 2: 0.05},
        2: {0: 0.05, 1: 0.10, 2: 0.85},
    }

    print("\n[SAVE] Saving regime model to regime_model.pkl...")
    try:
        model_data = {
            "spectral_model": spectral,
            "regime_anchors": regime_anchors,
            "transition_matrix": transition_matrix,
            "n_regimes": 3,
            "feature_names": [
                "time_horizon",
                "age",
                "crowd_size",
                "community_confidence",
                "complexity",
                "type",
                "activity_rate"
            ],
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
            "n_regimes": 3,
            "transition_matrix": transition_matrix,
            "regime_anchors": {
                str(k): {
                    "binary_default": float(v["binary_default"]),
                    "binary_mean": float(v["binary_mean"]),
                    "binary_std": float(v["binary_std"]),
                    "numeric_scale": float(v["numeric_scale"]),
                    "numeric_shift": float(v["numeric_shift"]),
                    "multiple_choice_confidence": float(v["multiple_choice_confidence"]),
                    "sample_size": v["sample_size"],
                    "avg_time_horizon_days": v["avg_time_horizon_days"],
                    "median_time_horizon_days": v["median_time_horizon_days"],
                    "avg_crowd_size": v["avg_crowd_size"],
                    "avg_complexity": v["avg_complexity"],
                    "confidence_weight": v["confidence_weight"],
                }
                for k, v in regime_anchors.items()
            },
            "training_questions": len(questions),
            "training_timestamp": datetime.now().isoformat(),
            "feature_names": [
                "time_horizon",
                "age",
                "crowd_size",
                "community_confidence",
                "complexity",
                "type",
                "activity_rate"
            ],
        }

        with open("regime_model_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("✓ Saved regime_model_summary.json")
    except Exception as e:
        print(f"ERROR saving regime_model_summary.json: {e}")
        return False

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE - 7-FEATURE MODEL WITH 3 REGIMES")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review regime characteristics above")
    print("2. Verify files: ls -lh regime_model.pkl regime_model_summary.json")
    print("3. Check summary: cat regime_model_summary.json")
    print("4. Update main.py with new extract_features() function")
    print("5. Test bot: python main.py --mode test_questions")
    print("6. Deploy: python main.py --mode tournament")

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
