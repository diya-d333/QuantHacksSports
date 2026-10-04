# Snowflake setup

## Existing resources
- Warehouse: GQH_WH
- Database: GQH_CRYPTO
- Schema: ML_DATA

The warehouse and database must already exist, and your
Snowflake role must have permission to use them.

## Setup order
1. Run setup_tables.sql in a Snowflake SQL worksheet.
2. Import regime_features_4h.csv into
   GQH_CRYPTO.ML_DATA.REGIME_FEATURES_4H.
3. Import regime_model_output/regime_predictions_4h.csv into
   GQH_CRYPTO.ML_DATA.REGIME_PREDICTIONS_4H.
4. Run set_signals.sql to create the joined data and signal
   views and run the validation queries.

Skip the imports if those files are already loaded.
Match CSV columns by header name and avoid duplicate imports.

## Pipeline
Tiger Data provides market and sentiment data.
Python builds features and trains the HMM locally.
Snowflake stores the features and predictions, generates
mean-reversion signals, and evaluates forward returns.

Backtrader currently recreates signals in Python.
Import automation and passing Snowflake signals directly
to Backtrader are still pending.

## Credentials
Keep credentials in a private .env file.
Never commit passwords or connection strings to GitHub.

## Snowflake HMM with ADX

BTC and ETH regime models train inside a Snowflake Python stored
procedure using hmmlearn. ADX is one of six model inputs.

- Training data: 2021–2024 (`REGIME_TRAIN`).
- Validation data: 2025 (`REGIME_VALIDATION`).
- The scaler and model fit only on training data.
- Validation predictions use observations available through each bar.
- The training state with the lowest mean ADX is labeled
  `RANGE_CANDIDATE`. This is a provisional interpretation.

### Execution order

With the input tables populated, run these SQL files in Snowflake:

1. `calculate_adx.sql` — create the ADX procedure.
2. Run `CALL GQH_CRYPTO.ML_DATA.CALCULATE_ADX();`
3. `setup_adx_features.sql` — join ADX to the existing features.
4. `train_regime_adx.sql` — create the training procedure.
5. Run `CALL GQH_CRYPTO.ML_DATA.TRAIN_REGIME_ADX();`
6. `set_signals_adx.sql` — create the joined data and signal views.

Use warehouse `GQH_WH`, database `GQH_CRYPTO`, and schema `ML_DATA`.

### Exported outputs

Files are saved in `regime_model_output/`:

- `regime_predictions_adx_4h.csv` — training and validation predictions.
- `mean_reversion_signals_adx_4h.csv` — validation rows with signal flags.
- `regime_state_statistics_adx.csv` — training-state statistics.

Teammates can use these exports without Snowflake access.

### Backtest integration

Join exports to the full OHLCV price timeline by asset and timestamp.
Signals become available at `BAR_END_TIME`; orders must execute
after the signal becomes available.

Do not use `FORWARD_RETURN_1` or `TARGET_READY` to decide entries.
They are evaluation fields.

The mean-reversion view implements large-drop, fear, and range filters.
A different strategy requires its own entry rules.

### Current limitations

Trained BTC and ETH models, scalers, feature order, and range-state
mappings are saved in the Snowflake stage:
GQH_CRYPTO.ML_DATA.REGIME_MODEL_ARTIFACTS/frozen_v1/

Training uses only 2021–2024. A separate inference step will load
these saved models for the final 2026 test without retraining.

The exports still need to be connected to the backtest.

## Integrated 2025 mean-reversion validation

Run: python examples/backtest/tiger_backtest.py

Entries require:
- A large BTC or ETH drop.
- A Snowflake RANGE_CANDIDATE label.
- Fear & Greed <= 25.
- Average XRP/SOL four-hour return >= -2%.
- Valid entry timing and available confirmation data.

The shared dataset contains 964 bars from May 18 through
December 31, 2025. It does not cover the full year.

Result: zero trades. XRP/SOL confirmation blocked all three
candidates that passed the drop, range, and sentiment conditions.
This provides no evidence of trading profitability.

Synthetic checks passed for next-open entry, a four-hour exit
on continuous bars, and a delayed exit across a trading gap.
Sizing and transaction costs have not yet been tested with fills.

Sentiment uses the previously configured 24-hour availability-delay
assumption; the database view was not reverified in this review.

Saved report:
backtest_results/mean_reversion_2025_validation.txt

2026 remains reserved for the final out-of-sample test.