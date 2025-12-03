# Alzheimer’s Risk Prediction (CIS 412 Final Project)

This project uses a global Alzheimer’s dataset (~74k records, 25 features) to:

- Explore risk factors (age, APOE4 gene, family history, lifestyle, etc.)
- Train and compare Logistic Regression and Random Forest models
- Tune thresholds for better recall on high-risk patients
- Deploy a simple Streamlit app to demo predictions

## Files

- `CIS412_Alzheimers_Project.ipynb` – full EDA + model training notebook
- `alz_streamlit_app.py` – Streamlit app script
- `alzheimers_best_model.joblib` – saved best pipeline (Random Forest + preprocessing)
- `alz_feature_metadata.joblib` – metadata for the app (feature lists, etc.)

The original dataset is from Kaggle:  
<https://www.kaggle.com/datasets/ankushpanday1/alzheimers-prediction-dataset-global>

## How to run the app

```bash
pip install -r requirements.txt
streamlit run alz_streamlit_app.py
