import pandas as pd

trades = pd.read_csv("historical_data.csv")

trades["dt_ist"] = pd.to_datetime(trades["Timestamp IST"], format="%d-%m-%Y %H:%M")
print("IST range:", trades["dt_ist"].min(), trades["dt_ist"].max())

trades["dt_epoch"] = pd.to_datetime(trades["Timestamp"], unit="ms")
print("epoch(ms) range:", trades["dt_epoch"].min(), trades["dt_epoch"].max())

print(trades[["Timestamp IST","dt_ist","Timestamp","dt_epoch"]].head())

# check consistency between the two
diff = (trades["dt_ist"] - trades["dt_epoch"]).dt.total_seconds()/3600
print("hours diff stats:\n", diff.describe())
