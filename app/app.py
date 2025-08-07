"""
Streamlit application for predicting bridge condition ratings.

This app has dropdown menus for categorical variables and
descriptive labels with units for numeric fields. When fields are
left blank, median imputation and missing flags are handled in
processing.py.
"""

import streamlit as st
import joblib
import os

from processing import transform_data  # type: ignore

DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(DIR, "bridge_risk_model.joblib")
COLUMNS_PATH = os.path.join(DIR, "columns.joblib")
BOXPLOT_PATH = os.path.join(DIR, "age_condition_boxplot.png")
SHAP_PATH = os.path.join(DIR, "shap_feature_importance.png")


@st.cache_resource
def load_model_and_columns():
    model = joblib.load(MODEL_PATH)
    columns_raw = joblib.load(COLUMNS_PATH)
    try:
        columns = list(columns_raw)
    except TypeError:
        columns = columns_raw
    return model, columns


model, model_columns = load_model_and_columns()
label_map = {0: "Poor", 1: "Fair", 2: "Good"}


def build_input_form():
    st.sidebar.header("Bridge Details")
    st.sidebar.markdown(
        "Fill in the details below. Units are noted in parentheses; "
        "leave fields blank to use training medians or trigger a missing flag."
    )

    form = st.sidebar.form("bridge_form")

    def make_text_input(label):
        return form.text_input(label, placeholder="")

    def make_selectbox(label, options):
        return form.selectbox(label, [""] + options, index=0)

    year_built = make_text_input("27 - Year Built (year)")
    adt = make_text_input("29 - Average Daily Traffic (vehicles/day)")
    main_span_material = make_selectbox("43A - Main Span Material", [
        "Concrete", "Steel", "Prestressed Concrete", "Timber", "Masonry", "Aluminum", "Other"
    ])
    main_span_design = make_selectbox("43B - Main Span Design", [
        "Girder", "Truss - Through", "Truss - Deck", "Slab", "Arch", "Suspension", "Cable-Stayed", "Culvert", "Other"
    ])
    num_spans = make_text_input("45 - Number of Spans in Main Unit")
    structure_length = make_text_input("49 - Structure Length (ft.)")
    bridge_age = make_text_input("Bridge Age (yr)")
    deck_area = make_text_input("CAT29 - Deck Area (sq. ft.)")
    year_reconstructed = make_text_input("106 - Year Reconstructed (year)")
    skew_angle = make_text_input("34 - Skew Angle (degrees)")
    max_span_length = make_text_input("48 - Length of Maximum Span (ft.)")
    roadway_width = make_text_input("51 - Bridge Roadway Width Curb to Curb (ft.)")
    inspection_freq = make_text_input("91 - Designated Inspection Frequency (months)")
    operating_rating = make_text_input("64 - Operating Rating (US tons)")
    inventory_rating = make_text_input("66 - Inventory Rating (US tons)")
    year_adt = make_text_input("30 - Year of Average Daily Traffic (year)")
    pct_truck = make_text_input("109 - Average Daily Truck Traffic (Percent of ADT)")
    future_adt = make_text_input("114 - Future Average Daily Traffic (vehicles/day)")
    year_future_adt = make_text_input("115 - Year of Future Average Daily Traffic (year)")
    project_cost = make_text_input("96 - Total Project Cost (USD)")
    computed_truck = make_text_input("Computed - Average Daily Truck Traffic (Volume)")
    avg_rel_humidity = make_text_input("Average Relative Humidity (%)")
    avg_temp = make_text_input("Average Temperature (°F)")
    max_temp = make_text_input("Maximum Temperature (°F)")
    min_temp = make_text_input("Minimum Temperature (°F)")
    mean_wind_speed = make_text_input("Mean Wind Speed (mph)")
    was_reconstructed = make_selectbox("Was_Reconstructed", ["No", "Yes"])

    submit = form.form_submit_button("Predict")

    if submit:
        def to_float_or_none(val: str):
            if val in (None, ""):
                return None
            try:
                return float(val)
            except ValueError:
                return None

        return {
            "27 - Year Built": to_float_or_none(year_built),
            "29 - Average Daily Traffic": to_float_or_none(adt),
            "43A - Main Span Material": main_span_material,
            "43B - Main Span Design": main_span_design,
            "45 - Number of Spans in Main Unit": to_float_or_none(num_spans),
            "49 - Structure Length (ft.)": to_float_or_none(structure_length),
            "Bridge Age (yr)": to_float_or_none(bridge_age),
            "CAT29 - Deck Area (sq. ft.)": to_float_or_none(deck_area),
            "106 - Year Reconstructed": to_float_or_none(year_reconstructed),
            "34 - Skew Angle (degrees)": to_float_or_none(skew_angle),
            "48 - Length of Maximum Span (ft.)": to_float_or_none(max_span_length),
            "51 - Bridge Roadway Width Curb to Curb (ft.)": to_float_or_none(roadway_width),
            "91 - Designated Inspection Frequency": to_float_or_none(inspection_freq),
            "64 - Operating Rating (US tons)": to_float_or_none(operating_rating),
            "66 - Inventory Rating (US tons)": to_float_or_none(inventory_rating),
            "30 - Year of Average Daily Traffic": to_float_or_none(year_adt),
            "109 - Average Daily Truck Traffic (Percent ADT)": to_float_or_none(pct_truck),
            "114 - Future Average Daily Traffic": to_float_or_none(future_adt),
            "115 - Year of Future Average Daily Traffic": to_float_or_none(year_future_adt),
            "96 - Total Project Cost": to_float_or_none(project_cost),
            "Computed - Average Daily Truck Traffic (Volume)": to_float_or_none(computed_truck),
            "Average Relative Humidity": to_float_or_none(avg_rel_humidity),
            "Average Temperature": to_float_or_none(avg_temp),
            "Maximum Temperature": to_float_or_none(max_temp),
            "Minimum Temperature": to_float_or_none(min_temp),
            "Mean Wind Speed": to_float_or_none(mean_wind_speed),
            "Was_Reconstructed": was_reconstructed,
        }

    return {}


def main():
    st.title("BridgeSense Condition Prediction")
    show_training_overview()

    user_input = build_input_form()
    if not user_input:
        return

    try:
        df_transformed = transform_data(user_input)
    except Exception as exc:
        st.error(f"Preprocessing failed: {exc}")
        return

    missing_cols = set(model_columns) - set(df_transformed.columns)
    for col in missing_cols:
        df_transformed[col] = 0
    df_transformed = df_transformed[model_columns]

    try:
        prediction = model.predict(df_transformed)[0]
        probas = model.predict_proba(df_transformed)[0]
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    predicted_label = label_map.get(int(prediction), "Unknown")
    proba_dict = {label_map[i]: round(float(probas[i]), 3) for i in range(len(probas))}

    st.write("### Prediction Result")
    st.success(f"The predicted condition is: **{predicted_label}**")
    st.write("### Class Probabilities")
    st.table(proba_dict)


def show_training_overview():
    st.markdown("---")
    st.header("Project Purpose and Training Results")
    st.write(
        """
        **Project Objective**

        This project's goal is to predict if a bridge in Georgia will receive 
        a **Poor**, **Fair**, or **Good** condition rating using available NBI 
        features such as traffic, environmental factors, and construction material. 

        Given that a poor rating indicates serious problems that may lead to 
        failure in the near future, this prediction provides critical insight 
        that can be used proactively to take active measures and avoid collapse, 
        rather than waiting for the scheduled times where bridges are inspected.

        **Key Training Insights**

        To train the model, we used features describing traffic,
        structural dimensions, age, and environmental conditions. An
        XGBoost classifier was chosen for its strong performance on
        imbalanced data. We also engineered interaction features, such
        as the product of bridge age and truck traffic volume, to
        capture nonlinear relationships.
        """
    )
    st.image(BOXPLOT_PATH, caption="Distribution of bridge age grouped by condition rating", use_container_width=True)
    st.image(SHAP_PATH, caption="SHAP Feature Importance for the Final Model", use_container_width=True)


if __name__ == "__main__":
    main()
