"""
Python script to encode + split data in preparation for analysis

Run this file before running model_analysis.py, as it saves
the models for use in the next file.

"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# -- Small function to load in the data ---
def load_data(filepath='../data/processed/training_data.csv'):
    df = pd.read_csv(filepath)

    print("Data Loaded!")
    return df

# --- Preparing the data for modeling ---
# We handle the data by encoding and splitting it
def prepare_data(df, feature_cols=['Driver_Race', 'Driver_Age_Group', 'Driver_Sex'], test_size=0.2, random_state=42):
    
    print("Preparing Data for Model Training...")

    df_original = df.copy()
    label_encoders = {}
    
    # Create a copy with only needed columns
    data = df[feature_cols + ['Stopped']].copy()
    
    # Encoding the categorical variables
    X_encoded = data[feature_cols].copy()
    for col in feature_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    X = X_encoded.values
    y = data['Stopped'].values
    
    # Maintaining class balance with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print("Data Encoded and Split!")
    return X_train, X_test, y_train, y_test, label_encoders, feature_cols, df_original

# --- Training both the Logistic Regression and Naive Bayes models ---
def train_models(X_train, y_train):

    print("Training Models...")
    # logistic regression
    lr_model = LogisticRegression(
        random_state=42, 
        max_iter=1000, 
        solver='lbfgs'
    )
    lr_model.fit(X_train, y_train)
    
    # naive bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    
    print("Models Trained!")
    return lr_model, nb_model

#
def save_models_and_data(lr_model, nb_model, X_train, X_test, y_train, y_test, label_encoders, feature_names, df_original, save_dir='../models/'):
    
    print("Saving Models to ../models/ directory ...")

    # Save logreg model
    with open(f'{save_dir}logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    
    # Save nb model
    with open(f'{save_dir}naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)

    training_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoders': label_encoders,
        'feature_names': feature_names,
        'df_original': df_original
    }

    # Save data and encoders
    with open(f'{save_dir}training_data.pkl', 'wb') as f:
        pickle.dump(training_data, f)

    print("Models Saved!")


# --- Main execution ---
if __name__ == "__main__":

    # Load merged data
    df = load_data('../data/processed/training_data.csv')
    
    # Preparing data
    X_train, X_test, y_train, y_test, label_encoders, feature_names, df_original = prepare_data(
        df, 
        feature_cols=['Driver_Race', 'Driver_Age_Group', 'Driver_Sex']
    )
    
    # Train logred and nb models
    lr_model, nb_model = train_models(X_train, y_train)
    
    # Saving everything for analysis
    save_models_and_data(
        lr_model, 
        nb_model, 
        X_train, 
        X_test, 
        y_train, 
        y_test,
        label_encoders, 
        feature_names, 
        df_original
    )

    # Done message
    print("Finished running model_training.py!")