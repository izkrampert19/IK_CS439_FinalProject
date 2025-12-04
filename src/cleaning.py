import pandas as pd
import os
import numpy as np

path = "../data/raw/StanfordOpenPolicing/ct_statewide_2020_04_01.csv"

df = pd.read_csv(path)

# print(df['subject_race'].value_counts())
print(df.shape)

# Drop all NaN values
# Standardize labels (Race, sex, age(group ages?))
# Choose Values belonging to one specific year, if possible. If one specific year has less than 
    # 300,000 stops, don't use it
# Final columns I want in the stop DF: 
    # Driver_Race
    # Driver_Age_Group
    # Driver_Sex
    # State
# Randomly select ~300k samples per table, make sure sample is stratified and is not biased

# Don't forget -- need US census bureau for negative examples (demographics of individuals NOT stopped)
# For race, age, and sex distributions of general populaion