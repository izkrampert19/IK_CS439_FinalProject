import pandas as pd
import os
import glob

PATH = "../data/raw/StanfordOpenPolicing/co_statewide_2020_04_01.csv"
YEAR = 2014
SAMPLE_SIZE = 300_000

"""df = pd.read_csv(PATH)
print(df['subject_sex'].value_counts())"""


# Function to group ages
def age_group(age):
    try:
        age = int(age)
    except:
        return None
    
    if age < 25:                # Young Adults
        return "Under 25"   
    elif age < 40:              # Adults
        return "25-39"
    elif age < 65:              # Middle Aged
        return "40-64"
    else:                       # Senior
        return "65+"

# Standardizing race categories -- NOT including "unkown" or "other"
RACE_MAP = {
    "white": "White",
    "black": "Black",
    "asian/pacific islander": "Asian/Pacific Islander"
}

# Standardizing sex categories
# Note: "male" or "female", but in case of different entries include "m" and "f"
SEX_MAP = {
    "male": "Male",
    "female": "Female",
    "m": "Male",
    "f": "Female"
}


""" CLEANING SCRIPT BELOW """

all_states_df = []

for file_path in glob.glob(PATH):
    state = os.path.basename(file_path).split("_")[0].upper()

    print(f"\nProcessing {state}")

    # Load in CSV
    df = pd.read_csv(file_path)

    # Drop null values for key fields
    df = df.dropna(subset=["subject_race", "subject_sex", "subject_age", "date"])

    # Obtain year from original "date" column
    df["Year"] = pd.to_datetime(df["date"], errors="coerce".dt.year)
    df = df[df["Year"] == YEAR]

    # Standardizing race and sex
    df["Driver_Race"] = (
        df["subject_race"]
        .str.lower()
        .map(RACE_MAP)
    )

    df["Driver_Sex"] = (
        df["subject_sex"]
        .str.lower()
        .map(SEX_MAP)
    )

    # Creating age groups
    df["Driver_Age_Group"] = df["subject_age"].apply(age_group)
    df = df.dropna(subset=["Driver_Age_Group"])

    # Add column for the state the driver belongs to
    df["State"] = state

    # Ensure that only the necessary columns are kept!!
    df = df[["Driver_Race", "Driver_Age_Group", "Driver_Sex", "State", "Year"]]

    # BEGIN STRATIFIED SAMPLING OF DATASET
    # Ensure that race groups remain proportional to prevent biases from forming
    if len(df) > SAMPLE_SIZE:
        df = df.groupby("Driver_Race").sample(
            frac=min(1, SAMPLE_SIZE / len(df)),
            random_state=42
        )

    all_states_df.append(df)

# Merging all the states
df_final = pd.concat(all_states_df, ignore_index=True)

df_final.to_csv("data/processed/cleaned.csv")