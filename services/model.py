import joblib
import pandas as pd
from fastapi import HTTPException

from config import MODEL_BUNDLE_PATH
from schemas.salary import SalaryPredictionInput

CATEGORICAL_COLUMNS = [
    "experience_level",
    "employment_type",
    "job_title_clean",
    "employee_residence",
    "company_location",
    "company_size",
]
NUMERIC_COLUMNS = ["work_year", "remote_ratio"]

try:
    bundle = joblib.load(MODEL_BUNDLE_PATH)
except FileNotFoundError as exc:
    raise RuntimeError(f"Model bundle not found at {MODEL_BUNDLE_PATH}") from exc

model = bundle["model"]
encoder = bundle["encoder"]
feature_names = bundle["feature_names"]


def preprocess(data: SalaryPredictionInput) -> pd.DataFrame:
    df = pd.DataFrame([data.model_dump()])

    df_encoded = encoder.transform(df[CATEGORICAL_COLUMNS])
    df_encoded = pd.DataFrame(
        df_encoded,
        columns=encoder.get_feature_names_out(CATEGORICAL_COLUMNS),
    )

    df_final = pd.concat([df[NUMERIC_COLUMNS].reset_index(drop=True), df_encoded], axis=1)

    for column in feature_names:
        if column not in df_final.columns:
            df_final[column] = 0

    return df_final[feature_names]


def predict_salary(data: SalaryPredictionInput) -> float:
    try:
        input_df = preprocess(data)
        prediction = model.predict(input_df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    return round(float(prediction[0]), 2)
