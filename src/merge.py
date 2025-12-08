import pandas as pd
import numpy as np

# Create mappings for demographic codes used in CSV
# Further info about KEY found in "sc-est2019-alldata5_KEY.pdf"
CENSUS_MAPPINGS = {
    'sex': {
        # 0 will eventually be excluded
        0: 'Total',
        1: 'Male', 
        2: 'Female'
    },
    'origin': {
        # 0 will eventually be excluded
        0: 'Total',
        1: 'Not Hispanic',
        2: 'Hispanic'
    },
    'race': {
        # As stated in sc-est2019-alldata_KEY.pdf
        # These do not match up with the Stanford Open Data currently, but this will be fixed later
        1: 'White Alone or in Combination',
        2: 'Black Alone or in Combination',
        3: 'American Indian and Alaska Native Alone or in Combination',
        4: 'Asian Alone or in Combination',
        5: 'Native Hawaiian and Other Pacific Islander Alone or in Combination'
    }
}


# --- Load the cleaned data of people who were stopped ---
# Returns a dataframe called "df_stopped" that adds a "Stopped" column.
def load_SOP_data(filepath='.../data/processed/cleaned.csv'):
    df_stopped = pd.read_csv(filepath)

    # Add indicator for stopped entries
    df_stopped['Stopped'] = 1
    return df_stopped

# --- Load the census data for our baseline ---
# Returns a dataframe called "df_census".
def load_USBC_data(filepath='.../data/processed/sc-est2019-alldata5_TRIMMED.csv'):
    df_census = pd.read_csv(filepath)

    # Mapping the numeric codes to the labels
    df_census['Sex_Label'] = df_census['SEX'].map(CENSUS_MAPPINGS['sex'])
    df_census['Race_Label'] = df_census['RACE'].map(CENSUS_MAPPINGS['race'])
    df_census['Origin_Label'] = df_census['ORIGIN'].map(CENSUS_MAPPINGS['origin'])

    # Filter out rows w/ aggregated data
    # SEX --> Male or Female only (Exclude 0, which is Total)
    # RACE --> Valid categories only
    # ORIGIN --> Hispanic or Not Hispanic (Exclude 0, which is Total)
    df_census = df_census[
        (df_census['SEX'].isin([1,2])) & (df_census['RACE'].isin([1,2,3,4,5])) & (df_census['ORIGIN'].isin([1,2]))
    ].copy()

    return df_census

# --- Create age groups according to cleaned data ---
def create_age_groups(age):
    if pd.isna(age): 
        return None
    try:
        age = int(age)
    except:
        return None
    
    if age < 25:
        return "Under 25"
    elif age < 40:
        return "25-39"
    elif age < 65:
        return "40-64"
    else:
        return "65+"

# --- Mapping the baseline data to the stop data ---
# Because the Stanford Open Project classifies "Hispanic" as a racial category while 
# the US Census Bureau does not, automatically mark any race of hispanic origin from
# USCB set as "Hispanic", regardless of race.
def map_USCB_SOP(census_race, origin_label):

    if origin_label == 'Hispanic':
        return 'Hispanic'
    
    race_mapping = {
        'White Alone or in Combination': 'White',
        'Black Alone or in Combination': 'Black',
        'American Indian and Alaska Native Alone or in Combination': None,      # Excluded from SOP data
        'Asian Alone or in Combiation': 'Asian/Pacific Islander', 
        'Native Hawaiian and Other Pacific Islander Alone or in Combination': 'Asian/Pacific Islander'
    }

    return race_mapping.get(census_race)

# --- Obtaining stratified samples from the USCB data ---
def sample_USBC(df_census, n_samples=50000, random_state = 42):

    # Creating the age groups
    df_census['Age_Group'] = df_census['AGE'].apply(create_age_groups)

    # Mapping the race and origin categories from Census to Stanford
    df_census['Mapped_Race'] = df_census.apply(
        lambda row: map_USCB_SOP(row['Race_Label'], row['Origin_Label']),
        axis=1
    )

    # Removing the rows that do not have valid mappings
    df_census = df_census[
        df_census['Age_Group'].notna() & df_census['Mapped_Race'].notna()
    ].copy()

    # --- Creating the weighted sample ---
    sampled_rows = []
    total_population = df_census['POPESTIMATE2014'].sum()

    for _, row in df_census.iterrows():

        # Now, we ave to calculate how many individuals to sample from this particular demographic group
        prop = row['POPESTIMATE2014'] / total_population
        n_to_sample = max(1, int(prop * n_samples))

        for _ in range(n_to_sample):
            sampled_rows.append({
                'Driver_Race': row['Mapped_Race'],
                'Driver_Age_Group': row['Age_Group'],
                'Driver_Sex': row['Sex_Label'],
                'State': 'Census', # *********
                'Year': 2014,       # Every entry will have this...maybe remove
                'Stopped': 0
            })

    df_baseline = pd.DataFrame(sampled_rows)

    # !! If we somehow oversample, we reduce to the exact size by cutting random entries.
    if len(df_baseline) > n_samples:

        df_baseline = df_baseline.sample(
            n=n_samples, 
            random_state=random_state
        )

    return df_baseline

# --- Merging the datasets ---
def merge_datasets(df_stopped, df_baseline):

    # Make sure both have columns in this order
    columns = ['Driver_Race', 'Driver_Age_Group', 'Driver_Sex', 'State', 'Year', 'Stopped']
    df_stopped = df_stopped[columns]
    df_baseline = df_baseline[columns]

    df_merged = pd.concat([df_stopped, df_baseline], ignore_index=True)

    return df_merged


# --- Main execition ---
# Creates a merged dataset to train the Logistic Regression and Naive Bayes Models
if __name__ == "__main__":

    # Loading in our processed datasets
    df_stopped = load_SOP_data('../data/processed/cleaned.csv')
    df_census = load_USBC_data('../data/processed/sc-est2019-alldata5_TRIMMED.csv')

    # Getting our sample of individuals who weren't stopped / our baseline
    df_baseline = sample_USBC(df_census, n_samples=50000)

    # Merging our datasets into one
    df_merged = merge_datasets(df_stopped, df_baseline)

    # Finally, it's ready to use for training -- save to CSV
    df_merged.to_csv('../data/processed/merged_data.csv', index=False)
