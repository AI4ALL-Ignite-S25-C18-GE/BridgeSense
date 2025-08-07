"""
Processing module for BridgeSense Streamlit application.

This module loads the list of expected model columns and defines
`transform_data` for preprocessing user input. Further handles missing
values by imputing medians or setting missing flags for specific
features. It also computes engineered features (interaction terms) to
mirror the training pipeline.
"""

import pandas as pd
import joblib

# Try to load the columns used in model training from 'columns.joblib'.
# If the file is missing, model_columns will remain empty and
# transform_data will raise an error when called.
try:
    # Load model columns from joblib and convert to a plain list.  If the
    # saved object is a pandas Index or other iterable, casting to list
    # prevents ambiguous truth value errors when checking emptiness.
    model_columns = list(joblib.load("app/columns.joblib"))
except FileNotFoundError:
    model_columns = []


# Features that have corresponding missing flag columns in the model.
# When any of these features is missing from user input, its value
# will be imputed and the flag set to 1.
FEATURES_WITH_FLAGS = {
    "64 - Operating Rating (US tons)": "64 - Operating Rating (US tons)_missing_flag",
    "66 - Inventory Rating (US tons)": "66 - Inventory Rating (US tons)_missing_flag",
    "109 - Average Daily Truck Traffic (Percent ADT)": "109 - Average Daily Truck Traffic (Percent ADT)_missing_flag",
    "Computed - Average Daily Truck Traffic (Volume)": "Computed - Average Daily Truck Traffic (Volume)_missing_flag",
}


# Default median values for numeric variables.  These are used for
# imputation when the user leaves a field blank.  You should update
# these values to match the medians of your training data.  For
# categorical variables, leave the value as an empty string.
MEDIAN_VALUES = {
    'Year': 2007.0,
    '27 - Year Built': 1970.0, 
    '29 - Average Daily Traffic': 1590.0, 
    '45 - Number of Spans in Main Unit': 3.0, 
    '49 - Structure Length (ft.)': 71.9, 
    'Bridge Age (yr)': 34.0, 
    'CAT29 - Deck Area (sq. ft.)': 2419.4, 
    '106 - Year Reconstructed': 0.0, 
    '34 - Skew Angle (degrees)': 0.0, 
    '48 - Length of Maximum Span (ft.)': 25.9, 
    '51 - Bridge Roadway Width Curb to Curb (ft.)': 24.0, 
    '91 - Designated Inspection Frequency': 24.0, 
    '64 - Operating Rating (US tons)': 47.6, 
    '66 - Inventory Rating (US tons)': 27.0, 
    '30 - Year of Average Daily Traffic': 2005.0, 
    '109 - Average Daily Truck Traffic (Percent ADT)': 1.0, 
    '114 - Future Average Daily Traffic': 2325.0, 
    '115 - Year of Future Average Daily Traffic': 2025.0, 
    '96 - Total Project Cost': 30.0, 
    'Computed - Average Daily Truck Traffic (Volume)': 24.0, 
    'Average Relative Humidity': 76.0, 
    'Average Temperature': 17.3, 
    'Maximum Temperature': 38.3, 
    'Minimum Temperature': -7.1, 
    'Mean Wind Speed': 1.0, 
    'Was_Reconstructed': 1.0, 
    '64 - Operating Rating (US tons)_missing_flag': 0.0, 
    '66 - Inventory Rating (US tons)_missing_flag': 0.0, 
    '109 - Average Daily Truck Traffic (Percent ADT)_missing_flag': 0.0, 
    '96 - Total Project Cost_missing_flag': 0.0, 
    'Computed - Average Daily Truck Traffic (Volume)_missing_flag': 0.0
}


def transform_data(input_data: dict) -> pd.DataFrame:
    """
    Transform raw user input into a DataFrame aligned with model
    expectations.

    Parameters:
    input_data : dict
        Dictionary of user inputs where keys are feature names and values
        may be strings, numeric values, or blank (None or empty string).

    Returns:
    pd.DataFrame
        A single‑row DataFrame with engineered features and one‑hot
        encoded categorical variables, aligned with the order of
        `model_columns`.
    """
    # Ensure model_columns is a non-empty list.  Evaluating a pandas
    # Index directly in a boolean context raises an error, so check its
    # length explicitly.
    if model_columns is None or len(model_columns) == 0:
        raise FileNotFoundError(
            "Model columns not loaded. Ensure the file with filepath 'app/columns.joblib' is present."
        )

    # Copy the input to avoid mutating the original dictionary.
    row = input_data.copy()

    # Initialize missing flags to zero.
    for feature, flag in FEATURES_WITH_FLAGS.items():
        row[flag] = 0

    # Handle missing values for features with flags: set flag and impute.
    for feature, flag in FEATURES_WITH_FLAGS.items():
        value = row.get(feature)
        if value in (None, "", " ", "NaN"):
            row[feature] = MEDIAN_VALUES.get(feature, 0)
            row[flag] = 1

    # Impute other numeric features without flags when blank.
    for feature, median in MEDIAN_VALUES.items():
        # Skip features already handled with flags.
        if feature in FEATURES_WITH_FLAGS:
            continue
        value = row.get(feature)
        if value in (None, "", " ", "NaN"):
            row[feature] = median

    # Create DataFrame from single row.
    df = pd.DataFrame([row])

    # Compute engineered features used during model training.
    age = df.get("Bridge Age (yr)", 0)
    traffic_volume = df.get(
        "Computed - Average Daily Truck Traffic (Volume)", 0
    )
    operating_rating = df.get("64 - Operating Rating (US tons)", 0)

    df["Age_x_Traffic"] = age * traffic_volume
    df["squared"] = age ** 2
    df["Age_x_OperatingRating"] = age * operating_rating

    # One‑hot encode categorical variables.
    df_encoded = pd.get_dummies(df)

    # Align DataFrame with model columns, filling missing columns with zeros.
    # Cast model_columns to a list to avoid ambiguous truth-value evaluation if
    # it happens to be a pandas Index.  This ensures reindex receives a
    # simple list of column names.
    columns_list = list(model_columns)
    df_final = df_encoded.reindex(columns=columns_list, fill_value=0)


    return df_final
