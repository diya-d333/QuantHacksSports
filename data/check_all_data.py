import pandas as pd

assets = ["BTC", "ETH", "XRP", "SOL"]

for asset in assets:
    file = f"data/raw_data/{asset}_4h.parquet"

    df = pd.read_parquet(file)

    print("=" * 50)
    print(asset)
    print("=" * 50)

    print("Number of candles:", len(df))
    print("First timestamp:", df.index.min())
    print("Last timestamp:", df.index.max())
    print("Duplicate timestamps:", df.index.duplicated().sum())
    print()