import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# -------------------------------------------------------------------
# Config: file names
# -------------------------------------------------------------------
DATA_CSV   = "alzheimers_prediction_dataset.csv"
MODEL_FILE = "alzheimers_best_model.joblib"
META_FILE  = "alz_feature_metadata.joblib"


# -------------------------------------------------------------------
# Load existing model OR retrain locally if loading fails
# -------------------------------------------------------------------
# @st.cache_resource
def load_or_train_model():
    try:
        model = joblib.load(MODEL_FILE)
        meta  = joblib.load(META_FILE)
        return model, meta
    except Exception as e:
        print("Failed to load saved model, retraining locally. Error:", e)

        # 1. Load data
        df = pd.read_csv(DATA_CSV)

        # 2. Clean / rename columns to match the notebook
        df2 = df.rename(columns={
            "Alzheimer’s Diagnosis": "alz_dx",
            "Genetic Risk Factor (APOE-ε4 allele)": "APOE4",
            "Urban vs Rural Living": "UrbanRural",
            "Physical Activity Level": "PhysicalActivity",
            "Family History of Alzheimer’s": "FamHist",
        })

        # 3. Target + features
        y = (
            df2["alz_dx"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"no": 0, "yes": 1})
        )
        X = df2.drop(columns=["alz_dx"])

        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # 4. Preprocess + RandomForest (your best model)
        preprocess = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ]
        )

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )

        pipe = Pipeline(steps=[
            ("prep", preprocess),
            ("clf", rf),
        ])

        pipe.fit(X_train, y_train)

        # 5. Save new artifacts using THIS sklearn version
        feature_metadata = {
            "feature_columns": X.columns.tolist(),
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "target_name": "alz_dx",
            "best_model_name": "Random Forest",
        }

        joblib.dump(pipe, MODEL_FILE)
        joblib.dump(feature_metadata, META_FILE)

        return pipe, feature_metadata


# Actually load (or retrain once)
model, meta = load_or_train_model()

feature_cols    = meta["feature_columns"]
num_cols        = meta["numeric_columns"]
cat_cols        = meta["categorical_columns"]
best_model_name = meta["best_model_name"]


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Alzheimer's Prediction Demo", page_icon="🧠")

st.title("Alzheimer's Risk Prediction")
st.markdown(
    "This demo uses a machine-learning model trained on a **74k-row global dataset** "
    "to estimate the probability that a person would receive an Alzheimer's diagnosis "
    "given their demographic, lifestyle, and medical characteristics."
)

st.sidebar.header("Patient profile")

# -------------------------
# Build input dictionary
# -------------------------
inputs = {}

# Numeric features – sliders with sensible ranges
for col in num_cols:
    if col == "Age":
        inputs[col] = st.sidebar.slider("Age (years)", 50, 100, 70)
    elif col == "BMI":
        inputs[col] = st.sidebar.slider("Body Mass Index (BMI)", 15.0, 40.0, 27.0, 0.1)
    elif col == "Education Level":
        inputs[col] = st.sidebar.slider("Education level (years)", 0, 20, 12)
    elif col == "Cognitive Test Score":
        inputs[col] = st.sidebar.slider("Cognitive test score", 0, 100, 65)
    else:
        val = st.sidebar.number_input(col, value=0.0)
        inputs[col] = float(val)

# Key categorical features
inputs["Gender"]           = st.sidebar.selectbox("Gender", ["Male", "Female"])
inputs["APOE4"]            = st.sidebar.selectbox("Genetic risk (APOE-ε4 allele)", ["No", "Yes"])
inputs["FamHist"]          = st.sidebar.selectbox("Family history of Alzheimer's", ["No", "Yes"])
inputs["Smoking Status"]   = st.sidebar.selectbox("Smoking status", ["Never", "Former", "Current"])
inputs["PhysicalActivity"] = st.sidebar.selectbox("Physical activity level", ["Low", "Medium", "High"])
inputs["UrbanRural"]       = st.sidebar.selectbox("Urban vs Rural living", ["Urban", "Rural"])

# Remaining categoricals – reasonable options / defaults
if "Country" in cat_cols:
    inputs["Country"] = st.sidebar.selectbox(
        "Country",
        [
            "USA", "India", "Brazil", "China", "France", "Germany", "Italy", "Spain",
            "Japan", "Mexico", "Canada", "Sweden", "Norway", "South Africa",
            "Russia", "Argentina", "Australia", "UK",
        ],
        index=0,
    )

if "Diabetes" in cat_cols:
    inputs["Diabetes"] = st.sidebar.selectbox("Diabetes", ["No", "Yes"])

if "Hypertension" in cat_cols:
    inputs["Hypertension"] = st.sidebar.selectbox("Hypertension", ["No", "Yes"])

if "Cholesterol Level" in cat_cols:
    inputs["Cholesterol Level"] = st.sidebar.selectbox(
        "Cholesterol level", ["Normal", "Borderline", "High"]
    )

if "Alcohol Consumption" in cat_cols:
    inputs["Alcohol Consumption"] = st.sidebar.selectbox(
        "Alcohol consumption", ["Never", "Occasionally", "Regularly"]
    )

if "Dietary Habits" in cat_cols:
    inputs["Dietary Habits"] = st.sidebar.selectbox(
        "Dietary habits", ["Healthy", "Average", "Unhealthy"]
    )

if "Sleep Quality" in cat_cols:
    inputs["Sleep Quality"] = st.sidebar.selectbox(
        "Sleep quality", ["Poor", "Average", "Good"]
    )

if "Depression Level" in cat_cols:
    inputs["Depression Level"] = st.sidebar.selectbox(
        "Depression level", ["Low", "Moderate", "High"]
    )

if "Social Engagement Level" in cat_cols:
    inputs["Social Engagement Level"] = st.sidebar.selectbox(
        "Social engagement level", ["Low", "Medium", "High"]
    )

if "Income Level" in cat_cols:
    inputs["Income Level"] = st.sidebar.selectbox(
        "Income level", ["Low", "Medium", "High"]
    )

if "Stress Levels" in cat_cols:
    inputs["Stress Levels"] = st.sidebar.selectbox(
        "Stress level", ["Low", "Medium", "High"]
    )

if "Employment Status" in cat_cols:
    inputs["Employment Status"] = st.sidebar.selectbox(
        "Employment status", ["Employed", "Unemployed", "Retired"]
    )

if "Marital Status" in cat_cols:
    inputs["Marital Status"] = st.sidebar.selectbox(
        "Marital status", ["Single", "Married", "Divorced", "Widowed"]
    )

# Fill any missing features with a generic default so schema matches
for col in feature_cols:
    if col not in inputs:
        inputs[col] = "Unknown"

# Turn into DataFrame in correct column order
input_df = pd.DataFrame([inputs])
input_df = input_df[feature_cols]

st.subheader("Current input")
st.dataframe(input_df)

# Threshold slider (ties back to your PR-curve step)
default_threshold = 0.50
thresh = st.slider("Decision threshold", 0.0, 1.0, float(default_threshold), 0.01)

if st.button("Predict Alzheimer's risk"):
    prob = model.predict_proba(input_df)[:, 1][0]
    pred = int(prob >= thresh)

    st.markdown("### Prediction")
    st.write(f"Estimated probability of Alzheimer's diagnosis: **{prob:.2%}**")
    st.write(f"Threshold: **{thresh:.2f}**")
    st.write(f"Predicted class: **{'Yes' if pred == 1 else 'No'}**")

    st.caption(
        f"Underlying model: {best_model_name} pipeline "
        "(StandardScaler + OneHotEncoder + classifier). "
        "On the held-out test set it achieved about **0.80 ROC-AUC**."
    )
