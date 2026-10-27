import pandas as pd

dataset = ("BRENT", "SQQQ", "USO")

for d in dataset:
    data = pd.read_csv(f"{d}/{d}.csv")

    data = data.round(2)

    data_no_na = data.dropna()

    diff = data_no_na.shape[0] - data.shape[0]

    if diff > 0:
        print(f"{d} has {diff} rows with NaN values")

    data_no_na.to_csv(f"{d}/{d}_clean.csv", index=False)