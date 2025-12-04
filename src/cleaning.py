import pandas as pd
import os
import numpy as np

path = "../data/raw/StanfordOpenPolicing/nc_statewide_2020_04_01.csv"

df = pd.read_csv(path, nrows=5)
print(df)

for col in df.columns:
    print(col)