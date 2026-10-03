from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "regime_features_4h.csv"
OUTPUT_DIR = PROJECT_ROOT / "regime_model_output"

# Regime inputs only.
# Sentiment/drop triggers remain separate.
# FORWARD_RETURN_1 must never be a model input.
MODEL_FEATURES = [
    "RETURN_6",
    "RETURN_30",
    "VOLATILITY_30",
    "SMA_20_50_SPREAD",
    "BB_WIDTH",
]


def sequence_starts(frame):
    """Reset after trading gaps or dataset boundaries."""
    gap = frame["TIME"].diff().ne(pd.Timedelta(hours=4))
    boundary = frame["DATASET_ROLE"].ne(
        frame["DATASET_ROLE"].shift()
    )
    
    starts = (gap | boundary).to_numpy(dtype=bool, copy=True)
    starts[0] = True
    return starts


def sequence_lengths(starts):
    positions = np.flatnonzero(starts)
    return np.diff(
        np.append(positions, len(starts))
    ).tolist()


def filter_states(model, observations, starts):
    """Infer each state using observations through that candle only."""
    emission_logs = np.column_stack([
        multivariate_normal.logpdf(
            observations,
            mean=model.means_[state],
            cov=model.covars_[state],
        )
        for state in range(model.n_components)
    ])

    probabilities = np.zeros(
        (len(observations), model.n_components)
    )
    previous = None

    for i in range(len(observations)):
        prior = (
            model.startprob_
            if starts[i]
            else previous @ model.transmat_
        )

        with np.errstate(divide="ignore"):
            log_weights = np.log(prior) + emission_logs[i]

        normalizer = logsumexp(log_weights)

        if not np.isfinite(normalizer):
            raise ValueError("Invalid state probabilities")

        previous = np.exp(log_weights - normalizer)
        probabilities[i] = previous

    return probabilities


def train_asset(data, asset):
    print(f"\nStarting {asset} training...", flush=True)

    frame = (
        data.loc[data["ASSET"].eq(asset)]
        .sort_values("TIME")
        .dropna(subset=MODEL_FEATURES)
        .reset_index(drop=True)
    )

    if frame.empty:
        raise ValueError(f"No usable features for {asset}")

    if not np.isfinite(
        frame[MODEL_FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{asset}: nonfinite model features")

    train_mask = frame["DATASET_ROLE"].eq("REGIME_TRAIN")
    training = frame.loc[train_mask].copy()

    if len(training) < 300:
        raise ValueError(f"{asset}: too few training rows")

    print(
        f"{asset}: {len(training):,} training rows",
        flush=True,
    )

    # Fit only on 2021–2024.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(training[MODEL_FEATURES])

    lengths = sequence_lengths(sequence_starts(training))

    best_model = None
    best_score = -np.inf

    for seed in [42, 43, 44]:
        print(f"{asset}: fitting seed {seed}...", flush=True)

        model = GaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=500,
            tol=0.01,
            min_covar=0.001,
            random_state=seed,
        )

        model.fit(x_train, lengths=lengths)

        history = list(model.monitor_.history)
        stable = (
            len(history) >= 2
            and abs(history[-1] - history[-2]) < 0.01
        )

        if not stable:
            print(f"{asset}: seed {seed} did not stabilize; skipped")
            continue

        # Choose initialization using training likelihood only.
        score = model.score(x_train, lengths=lengths)

        if np.isfinite(score) and score > best_score:
            best_model = model
            best_score = score

    if best_model is None:
        raise RuntimeError(
            f"{asset}: no stable model. "
            "Send the terminal output before proceeding."
        )

    # Frozen scaler/model for validation.
    x_all = scaler.transform(frame[MODEL_FEATURES])

    probabilities = filter_states(
        best_model,
        x_all,
        sequence_starts(frame),
    )

    frame["HMM_STATE"] = probabilities.argmax(axis=1)
    frame["STATE_CONFIDENCE"] = probabilities.max(axis=1)

    training_labels = frame.loc[train_mask]

    statistics = (
        training_labels.groupby("HMM_STATE")[MODEL_FEATURES]
        .mean()
    )

    statistics["TRAIN_ROWS"] = (
        training_labels.groupby("HMM_STATE").size()
    )

    if len(statistics) != 3:
        raise RuntimeError(
            f"{asset}: not all three states are represented"
        )

    # Provisional range candidate: weakest absolute trend.
    # This uses training features, never future returns.
    trend_positions = [
        MODEL_FEATURES.index("RETURN_30"),
        MODEL_FEATURES.index("SMA_20_50_SPREAD"),
    ]

    train_states = training_labels["HMM_STATE"].to_numpy()
    trend_scores = {}

    for state in range(3):
        state_rows = x_train[train_states == state]
        trend_scores[state] = np.abs(
            state_rows[:, trend_positions]
        ).mean()

    range_state = min(trend_scores, key=trend_scores.get)

    statistics["RANGE_CANDIDATE_SCORE"] = pd.Series(trend_scores)
    statistics["ASSET"] = asset

    frame["REGIME"] = np.where(
        frame["HMM_STATE"].eq(range_state),
        "RANGE_CANDIDATE",
        "OTHER",
    )

    frame["RANGE_PROBABILITY"] = probabilities[:, range_state]

    joblib.dump(
        {
            "model": best_model,
            "scaler": scaler,
            "features": MODEL_FEATURES,
            "range_candidate_state": int(range_state),
        },
        OUTPUT_DIR / f"{asset}_hmm.joblib",
    )

    print(f"\n{asset} training-state statistics:")
    print(statistics.to_string())

    print(f"\n{asset} label counts:")
    print(
        frame.groupby(["DATASET_ROLE", "REGIME"])
        .size()
        .to_string()
    )

    columns = [
        "TIME",
        "BAR_END_TIME",
        "ASSET",
        "DATASET_ROLE",
        "HMM_STATE",
        "REGIME",
        "STATE_CONFIDENCE",
        "RANGE_PROBABILITY",
    ]

    return frame[columns], statistics.reset_index()


def main():
    print("Starting regime-model script...", flush=True)
    print(f"Reading: {INPUT_FILE}", flush=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"CSV not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_FILE)

    for column in ["TIME", "BAR_END_TIME"]:
        data[column] = pd.to_datetime(data[column], utc=True)

    data = data.loc[
        data["DATASET_ROLE"].isin([
            "REGIME_TRAIN",
            "REGIME_VALIDATION",
        ])
    ].copy()

    if data.duplicated(["ASSET", "TIME"]).any():
        raise ValueError("Duplicate asset/timestamp rows")

    predictions = []
    statistics = []

    for asset in ["BTC", "ETH"]:
        asset_predictions, asset_statistics = train_asset(
            data, asset
        )
        predictions.append(asset_predictions)
        statistics.append(asset_statistics)

    pd.concat(predictions, ignore_index=True).to_csv(
        OUTPUT_DIR / "regime_predictions_4h.csv",
        index=False,
    )

    pd.concat(statistics, ignore_index=True).to_csv(
        OUTPUT_DIR / "regime_state_statistics.csv",
        index=False,
    )

    print(f"\nFinished. Outputs saved in: {OUTPUT_DIR}")
    print("Inspect RANGE_CANDIDATE before accepting it as range-bound.")


if __name__ == "__main__":
    main()