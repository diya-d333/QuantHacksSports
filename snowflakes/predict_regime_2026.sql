USE WAREHOUSE GQH_WH;
USE DATABASE GQH_CRYPTO;
USE SCHEMA ML_DATA;

CREATE OR REPLACE PROCEDURE
    GQH_CRYPTO.ML_DATA.PREDICT_REGIME_2026()
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
ARTIFACT_REPOSITORY =
    snowflake.snowpark.pypi_shared_repository
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
import pickle

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from snowflake.snowpark.types import (
    StructType, StructField, StringType,
    DoubleType, LongType, TimestampType,
    TimestampTimeZone,
)


EXPECTED_FEATURES = [
    "RETURN_6",
    "RETURN_30",
    "VOLATILITY_30",
    "SMA_20_50_SPREAD",
    "BB_WIDTH",
    "ADX_14",
]


def filter_states(model, observations, starts):
    """Use only observations available through the current candle."""
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
    rows = session.sql("""
        SELECT
            TIME,
            BAR_END_TIME,
            ASSET,
            DATASET_ROLE,
            RETURN_6,
            RETURN_30,
            VOLATILITY_30,
            SMA_20_50_SPREAD,
            BB_WIDTH,
            ADX_14
        FROM GQH_CRYPTO.ML_DATA.REGIME_FEATURES_ADX_2026_4H
        WHERE DATASET_ROLE = 'REGIME_OOS'
          AND FEATURES_READY
          AND ADX_14 IS NOT NULL
        ORDER BY ASSET, TIME
    """).collect()

    if not rows:
        raise ValueError("No model-ready 2026 features found")

    data = pd.DataFrame([row.as_dict() for row in rows])

    for column in ["TIME", "BAR_END_TIME"]:
        data[column] = pd.to_datetime(data[column], utc=True)

    if data.duplicated(["ASSET", "TIME"]).any():
        raise ValueError("Duplicate asset/timestamp features")

    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-10-01", tz="UTC")

    if not (
        (data["BAR_END_TIME"] >= start)
        & (data["BAR_END_TIME"] < end)
    ).all():
        raise ValueError("Features fall outside the OOS period")

    output = []
    summary = []

    for asset in ["BTC", "ETH"]:
        frame = (
            data.loc[data["ASSET"].eq(asset)]
            .sort_values("TIME")
            .reset_index(drop=True)
        )

        if frame.empty:
            raise ValueError(f"No model-ready rows for {asset}")

        artifact_path = (
            "@GQH_CRYPTO.ML_DATA.REGIME_MODEL_ARTIFACTS/"
            f"frozen_v1/{asset}_hmm_adx.pkl"
        )

        with session.file.get_stream(
            artifact_path,
            decompress=False,
        ) as stream:
            artifact = pickle.load(stream)

        if artifact["asset"] != asset:
            raise ValueError("Model asset does not match")

        features = list(artifact["features"])
        if features != EXPECTED_FEATURES:
            raise ValueError("Frozen model feature order differs")

        training_end = pd.to_datetime(
            artifact["training_end"], utc=True
        )
        if training_end >= start:
            raise ValueError("Model training overlaps the OOS period")

        model = artifact["model"]
        scaler = artifact["scaler"]
        range_state = int(artifact["range_candidate_state"])

        if not 0 <= range_state < model.n_components:
            raise ValueError("Invalid saved range state")

        values = frame[features].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{asset}: nonfinite features")

        # Apply the saved scaler. Never fit on 2026.
        observations = scaler.transform(frame[features])

        # Match the original inference policy:
        # reset at the period boundary and after trading gaps.
        starts = (
            frame["TIME"].diff()
            .ne(pd.Timedelta(hours=4))
            .to_numpy(dtype=bool, copy=True)
        )
        starts[0] = True

        probabilities = filter_states(
            model, observations, starts
        )
        states = probabilities.argmax(axis=1)

        for i, row in frame.iterrows():
            state = int(states[i])
            output.append((
                row["TIME"].to_pydatetime(),
                row["BAR_END_TIME"].to_pydatetime(),
                asset,
                "REGIME_OOS",
                state,
                (
                    "RANGE_CANDIDATE"
                    if state == range_state
                    else "OTHER"
                ),
                float(probabilities[i].max()),
                float(probabilities[i, range_state]),
                float(row["ADX_14"]),
            ))

        summary.append({
            "asset": asset,
            "prediction_rows": int(len(frame)),
            "model_version": "frozen_v1",
            "range_candidate_state": range_state,
            "retrained": False,
        })

    schema = StructType([
        StructField(
            "TIME", TimestampType(TimestampTimeZone.TZ)
        ),
        StructField(
            "BAR_END_TIME", TimestampType(TimestampTimeZone.TZ)
        ),
        StructField("ASSET", StringType()),
        StructField("DATASET_ROLE", StringType()),
        StructField("HMM_STATE", LongType()),
        StructField("REGIME", StringType()),
        StructField("STATE_CONFIDENCE", DoubleType()),
        StructField("RANGE_PROBABILITY", DoubleType()),
        StructField("ADX_14", DoubleType()),
    ])

    session.create_dataframe(
        output, schema=schema
    ).write.mode("overwrite").save_as_table(
        "GQH_CRYPTO.ML_DATA.REGIME_PREDICTIONS_ADX_2026_4H"
    )

    return summary
$$;