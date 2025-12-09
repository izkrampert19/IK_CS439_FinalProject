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
def load_SOP_data(filepath='../data/processed/cleaned.csv'):
    df_stopped = pd.read_csv(filepath)

    # Add indicator for stopped entries
    df_stopped['Stopped'] = 1
    return df_stopped

# --- Load the census data for our baseline ---
# Returns a dataframe called "df_census".
def load_USBC_data(filepath='../data/processed/sc-est2019-alldata5_TRIMMED.csv'):
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
# the US Census Bureau does not, we have to mark any race of Hispanic origin from the
# USCB set as "Hispanic", regardless of race.
def map_USCB_SOP(census_race, origin_label):

    if origin_label == 'Hispanic':
        return 'Hispanic'
    
    race_mapping = {
        'White Alone or in Combination': 'White',
        'Black Alone or in Combination': 'Black',
        'American Indian and Alaska Native Alone or in Combination': None,      # Excluded from SOP data
        'Asian Alone or in Combination': 'Asian/Pacific Islander', 
        'Native Hawaiian and Other Pacific Islander Alone or in Combination': 'Asian/Pacific Islander'
    }

    return race_mapping.get(census_race)

# --- Generating stratified, synthetic samples based the USCB data ---
# Because the USCB rows are totals and not individuals, we have to generate a synthetic dataset
# for our model 
def generate_sample_USBC(df_census_state, n_target, state_name):
    
    df = df_census_state.copy()

    # Create age groups and mapped race
    df['Age_Group'] = df['AGE'].apply(create_age_groups)
    df['Mapped_Race'] = df.apply(
        lambda row: map_USCB_SOP(row['Race_Label'], row['Origin_Label']),
        axis=1
    )

    # Dropping any rows that are invalid
    df = df[
        df['Age_Group'].notna() &
        df['Mapped_Race'].notna()
    ].copy()

    total_population = df['POPESTIMATE2014'].sum()

    rows = []

    for _, row in df.iterrows():

        prop = row['POPESTIMATE2014'] / total_population
        n_to_sample = max(1, int(prop * n_target))

        person_df = pd.DataFrame({
            'Driver_Race': [row['Mapped_Race']] * n_to_sample,
            'Driver_Age_Group': [row['Age_Group']] * n_to_sample,
            'Driver_Sex': [row['Sex_Label']] * n_to_sample,
            'State': [state_name] * n_to_sample,
            'Year':[2014] * n_to_sample,
            'Stopped': [0] * n_to_sample
        })
        rows.append(person_df)

    baseline = pd.concat(rows, ignore_index=True)

    # Trim if we accidentally overshoot
    if len(baseline) > n_target:
        baseline = baseline.sample(n=n_target, random_state=42)

    return baseline

# --- Merging the stopped and baseline datasets ---
def merge_datasets(df_stopped, df_baseline):

    cols = ['Driver_Race', 'Driver_Age_Group', 'Driver_Sex', 'State', 'Year', 'Stopped']

    df_s = df_stopped[cols].copy()
    df_b = df_baseline[cols].copy()

    df_merged = pd.concat([df_s, df_b], ignore_index=True)

    return df_merged

# --- Main execution ---
if __name__ == '__main__':

    # Load datasets in
    df_stopped = load_SOP_data()
    df_census = load_USBC_data()

    # To hold synthetic baselines for each state
    all_baselines = []

    states = df_stopped['State'].unique()

    for st in states:

       df_st = df_stopped[df_stopped['State']==st]
       n_stopped = len(df_st)
       
       df_census_state = df_census[df_census['NAME'].notna()]

       baseline_st = generate_sample_USBC(df_census_state, n_target=n_stopped, state_name=st)
       all_baselines.append(baseline_st)
    
    # combining all the baseline samples
    df_baseline = pd.concat(all_baselines, ignore_index=True)

    # finally, merging the stopped and baseline dfs
    df_merged = merge_datasets(df_stopped, df_baseline)

    # save to processed folder
    df_merged.to_csv('../data/processed/training_data.csv', index=False)

    # Print finished message
    print("Merged dataset saved to ../data/processed/training_data.csv")