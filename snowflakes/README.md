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