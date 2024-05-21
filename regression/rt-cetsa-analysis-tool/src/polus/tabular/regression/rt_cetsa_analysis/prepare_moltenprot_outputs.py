"""Preprocess pandas dataframmes in python."""

from pathlib import Path

import pandas as pd
from scipy.constants import convert_temperature

pd.options.display.precision = 1


inp_dir = Path(
    "/Users/antoinegerardin/RT-CETSA-Analysis/.data/final_outputs/moltenprot/",
)
params = "plate_(1-59)_moltenprot_params.csv"
values = "plate_(1-59)_moltenprot_curves.csv"

df = pd.read_csv(inp_dir / values)
df["Temperature"] = (
    df["Temperature"]
    .map(lambda temp: convert_temperature(temp, "Kelvin", "Celsius"))
    .round(1)
    .astype(str)
)
df["Temperature"] = "t_" + df["Temperature"]

print(list(df.columns))
print(df.index)
df = df.transpose()
print(df.head)

print(list(df.columns))

df.to_csv("transformed_curves.csv")
