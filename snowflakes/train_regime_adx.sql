USE WAREHOUSE GQH_WH;
USE DATABASE GQH_CRYPTO;
USE SCHEMA ML_DATA;

CREATE OR REPLACE PROCEDURE
    GQH_CRYPTO.ML_DATA.TRAIN_REGIME_ADX()
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = (
    'snowflake-snowpark-python',
    'hmmlearn==0.3.2',
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn'
)
HANDLER = 'main'
EXECUTE AS CALLER
AS
$$
import json
import io
import pickle
import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

from snowflake.snowpark.types import (
    StructType, StructField, StringType,
    DoubleType, LongType, TimestampType,
    TimestampTimeZone
)

FEATURES = [
    "RETURN_6",
    "RETURN_30",
    "VOLATILITY_30",
    "SMA_20_50_SPREAD",
    "BB_WIDTH",
    "ADX_14",
]


def sequence_starts(frame):
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
    # Forward filtering: each prediction uses observations
    # through that bar only, with frozen model parameters.
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
            weights = np.log(prior) + emission_logs[i]

        normalizer = logsumexp(weights)

        if not np.isfinite(normalizer):
            raise ValueError("Invalid state probabilities")

        previous = np.exp(weights - normalizer)
        probabilities[i] = previous

    return probabilities


def main(session):
    # Future returns and sentiment are excluded from model inputs.
    data = session.sql("""
        SELECT
            TIME, BAR_END_TIME, ASSET, DATASET_ROLE,
            RETURN_6, RETURN_30, VOLATILITY_30,
            SMA_20_50_SPREAD, BB_WIDTH, ADX_14
        FROM GQH_CRYPTO.ML_DATA.REGIME_FEATURES_ADX_4H
        WHERE FEATURES_READY
          AND ADX_14 IS NOT NULL
          AND DATASET_ROLE IN (
              'REGIME_TRAIN', 'REGIME_VALIDATION'
          )
        ORDER BY ASSET, TIME
    """).collect()

    data = pd.DataFrame([row.as_dict() for row in data])

    if data.empty:
        raise ValueError("No model-ready rows")

    for column in ["TIME", "BAR_END_TIME"]:
        data[column] = pd.to_datetime(data[column], utc=True)

    if data[["TIME", "BAR_END_TIME", "ASSET"]].isna().any().any():
        raise ValueError("Missing asset or timestamp")

    if data.duplicated(["ASSET", "TIME"]).any():
        raise ValueError("Duplicate asset/timestamp rows")

    if data[FEATURES].isna().any().any():
        raise ValueError("Missing model features")

    if not np.isfinite(
        data[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError("Nonfinite model features")

    # Confirm that dataset labels agree with the intended years.
    years = data["BAR_END_TIME"].dt.year
    invalid_roles = (
        data["DATASET_ROLE"].eq("REGIME_TRAIN")
        & ~years.between(2021, 2024)
    ) | (
        data["DATASET_ROLE"].eq("REGIME_VALIDATION")
        & years.ne(2025)
    )

    if invalid_roles.any():
        raise ValueError("Dataset role/year mismatch")

    prediction_rows = []
    statistics_rows = []
    summaries = []

    for asset in ["BTC", "ETH"]:
        frame = (
            data.loc[data["ASSET"].eq(asset)]
            .sort_values("TIME")
            .reset_index(drop=True)
        )

        if frame.empty:
            raise ValueError(f"No rows for {asset}")

        train_mask = frame["DATASET_ROLE"].eq("REGIME_TRAIN")
        training = frame.loc[train_mask].copy()

        if len(training) < 300:
            raise ValueError(f"{asset}: too few training rows")

        scaler = StandardScaler()
        x_train = scaler.fit_transform(training[FEATURES])
        lengths = sequence_lengths(sequence_starts(training))

        best_model = None
        best_score = -np.inf
        best_seed = None

        for seed in [42, 43, 44]:
            model = GaussianHMM(
                n_components=3,
                covariance_type="diag",
                n_iter=2000,
                tol=0.01,
                min_covar=0.001,
                random_state=seed,
            )

            model.fit(x_train, lengths=lengths)

            history = list(model.monitor_.history)
            stable = (
                len(history) >= 2
                and np.isfinite(history[-2:]).all()
                and abs(history[-1] - history[-2])
                    / len(training) < 0.00001
            )
            if not stable:
                if asset == "ETH":
                    summaries.append({
                        "seed": seed,
                        "iterations": model.monitor_.iter,
                        "last_scores": history[-5:],
                    })
                continue

            score = model.score(x_train, lengths=lengths)

            if np.isfinite(score) and score > best_score:
                best_model = model
                best_score = float(score)
                best_seed = seed

        if best_model is None:
            raise RuntimeError(
                f"{asset}: no stable model; diagnostics={summaries}"
            )

        observations = scaler.transform(frame[FEATURES])

        probabilities = filter_states(
            best_model,
            observations,
            sequence_starts(frame),
        )

        states = probabilities.argmax(axis=1)
        training_states = states[train_mask.to_numpy()]

        if len(np.unique(training_states)) != 3:
            raise RuntimeError(
                f"{asset}: not all three training states represented"
            )

        # Select the provisional range state using training ADX only.
        mean_adx = {
            state: float(
                training.loc[
                    training_states == state, "ADX_14"
                ].mean()
            )
            for state in range(3)
        }

        range_state = min(mean_adx, key=mean_adx.get)
        artifact = {
            "model": best_model,
            "scaler": scaler,
            "features": list(FEATURES),
            "asset": asset,
            "range_candidate_state": int(range_state),
            "selected_seed": int(best_seed),
            "training_start": training["BAR_END_TIME"].min().isoformat(),
            "training_end": training["BAR_END_TIME"].max().isoformat(),
            "sequence_reset": "4-hour gap or dataset-role boundary",
        }

        model_bytes = io.BytesIO(
            pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
        )

        session.file.put_stream(
            model_bytes,
            f"@GQH_CRYPTO.ML_DATA.REGIME_MODEL_ARTIFACTS/"
            f"frozen_v1/{asset}_hmm_adx.pkl",
            auto_compress=False,
            overwrite=False,
        )

        for state in range(3):
            subset = training.loc[training_states == state]

            statistics_rows.append((
                asset,
                int(state),
                int(len(subset)),
                mean_adx[state],
                float(subset["RETURN_30"].mean()),
                float(subset["VOLATILITY_30"].mean()),
                float(subset["SMA_20_50_SPREAD"].mean()),
                "RANGE_CANDIDATE"
                if state == range_state else "OTHER",
                int(best_seed),
                best_score,
            ))

        for i, row in frame.iterrows():
            state = int(states[i])

            prediction_rows.append((
                row["TIME"].to_pydatetime(),
                row["BAR_END_TIME"].to_pydatetime(),
                asset,
                str(row["DATASET_ROLE"]),
                state,
                "RANGE_CANDIDATE"
                if state == range_state else "OTHER",
                float(probabilities[i].max()),
                float(probabilities[i, range_state]),
                float(row["ADX_14"]),
            ))

        summaries.append({
            "asset": asset,
            "training_rows": int(len(training)),
            "validation_rows": int((~train_mask).sum()),
            "selected_seed": int(best_seed),
            "range_candidate_state": int(range_state),
        })

    timestamp_type = TimestampType(TimestampTimeZone.TZ)

    prediction_schema = StructType([
        StructField("TIME", timestamp_type),
        StructField("BAR_END_TIME", timestamp_type),
        StructField("ASSET", StringType()),
        StructField("DATASET_ROLE", StringType()),
        StructField("HMM_STATE", LongType()),
        StructField("REGIME", StringType()),
        StructField("STATE_CONFIDENCE", DoubleType()),
        StructField("RANGE_PROBABILITY", DoubleType()),
        StructField("ADX_14", DoubleType()),
    ])

    statistics_schema = StructType([
        StructField("ASSET", StringType()),
        StructField("HMM_STATE", LongType()),
        StructField("TRAIN_ROWS", LongType()),
        StructField("MEAN_ADX_14", DoubleType()),
        StructField("MEAN_RETURN_30", DoubleType()),
        StructField("MEAN_VOLATILITY_30", DoubleType()),
        StructField("MEAN_SMA_20_50_SPREAD", DoubleType()),
        StructField("REGIME", StringType()),
        StructField("SELECTED_SEED", LongType()),
        StructField("TRAIN_LOG_LIKELIHOOD", DoubleType()),
    ])

    session.create_dataframe(
        prediction_rows,
        schema=prediction_schema,
    ).write.mode("overwrite").save_as_table(
        "GQH_CRYPTO.ML_DATA.REGIME_PREDICTIONS_ADX_4H"
    )

    session.create_dataframe(
        statistics_rows,
        schema=statistics_schema,
    ).write.mode("overwrite").save_as_table(
        "GQH_CRYPTO.ML_DATA.REGIME_STATE_STATISTICS_ADX"
    )

    return json.dumps(summaries)
$$;