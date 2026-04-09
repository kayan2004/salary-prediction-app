from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ds_salaries.csv"
MODEL_PATH = BASE_DIR / "models" / "salary_model_bundle.pkl"
METRICS_PATH = BASE_DIR / "models" / "training_metrics.json"

TOP_N_JOB_TITLES = 25
TARGET_COLUMN = "salary_in_usd"
CATEGORICAL_COLUMNS = [
    "experience_level",
    "employment_type",
    "job_title_clean",
    "employee_residence",
    "company_location",
    "company_size",
]
NUMERIC_COLUMNS = ["work_year", "remote_ratio"]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()

    top_titles = cleaned_df["job_title"].value_counts().nlargest(TOP_N_JOB_TITLES).index
    cleaned_df["job_title_clean"] = cleaned_df["job_title"].where(
        cleaned_df["job_title"].isin(top_titles),
        "Other",
    )

    return cleaned_df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, OneHotEncoder]:
    x_categorical = df[CATEGORICAL_COLUMNS]
    x_numeric = df[NUMERIC_COLUMNS]
    y = df[TARGET_COLUMN]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_categorical_encoded = encoder.fit_transform(x_categorical)
    x_categorical_encoded = pd.DataFrame(
        x_categorical_encoded,
        columns=encoder.get_feature_names_out(CATEGORICAL_COLUMNS),
        index=df.index,
    )

    x = pd.concat([x_numeric, x_categorical_encoded], axis=1)
    return x, y, encoder


def train_model(x: pd.DataFrame, y: pd.Series) -> tuple[DecisionTreeRegressor, dict[str, float]]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = DecisionTreeRegressor(random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count": int(x.shape[1]),
    }
    return model, metrics


def save_artifacts(
    model: DecisionTreeRegressor,
    encoder: OneHotEncoder,
    feature_names: list[str],
    metrics: dict[str, float],
) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "encoder": encoder,
            "feature_names": feature_names,
        },
        MODEL_PATH,
    )

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    df = load_dataset(DATA_PATH)
    cleaned_df = clean_dataset(df)
    x, y, encoder = build_features(cleaned_df)
    model, metrics = train_model(x, y)
    save_artifacts(model, encoder, x.columns.tolist(), metrics)

    print(f"Training completed with {len(cleaned_df)} cleaned rows.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
