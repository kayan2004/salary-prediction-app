from __future__ import annotations

from itertools import product
from pathlib import Path
import random
import sys

import pandas as pd
import requests
from pydantic import ValidationError

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from schemas.salary import (
    COMPANY_SIZE_VALUES,
    EMPLOYMENT_TYPE_VALUES,
    EXPERIENCE_LEVEL_VALUES,
    REMOTE_RATIO_VALUES,
    SalaryPredictionInput,
    WORK_YEAR_VALUES,
)

API_URL = "http://127.0.0.1:8000/predict_salary"
OUTPUT_PATH = BASE_DIR / "data" / "predictions.csv"
MAX_REQUESTS = 200
PROGRESS_EVERY = 25
RANDOM_SEED = 42

# A focused subset keeps runtime reasonable while still covering the space well.
JOB_TITLES_TO_QUERY = [
    "Data Scientist",
    "Data Analyst",
    "Data Engineer",
    "Machine Learning Engineer",
    "Research Scientist",
    "Other",
]
COUNTRY_PAIRS = [
    ("US", "US"),
    ("GB", "GB"),
    ("DE", "DE"),
    ("IN", "US"),
    ("CA", "US"),
]


def generate_inputs() -> list[SalaryPredictionInput]:
    payloads: list[SalaryPredictionInput] = []

    for (
        work_year,
        remote_ratio,
        experience_level,
        employment_type,
        company_size,
        job_title_clean,
        (employee_residence, company_location),
    ) in product(
        WORK_YEAR_VALUES,
        REMOTE_RATIO_VALUES,
        EXPERIENCE_LEVEL_VALUES,
        EMPLOYMENT_TYPE_VALUES,
        COMPANY_SIZE_VALUES,
        JOB_TITLES_TO_QUERY,
        COUNTRY_PAIRS,
    ):
        try:
            payloads.append(
                SalaryPredictionInput(
                    work_year=work_year,
                    remote_ratio=remote_ratio,
                    experience_level=experience_level,
                    employment_type=employment_type,
                    job_title_clean=job_title_clean,
                    employee_residence=employee_residence,
                    company_location=company_location,
                    company_size=company_size,
                )
            )
        except ValidationError:
            continue

    if len(payloads) <= MAX_REQUESTS:
        return payloads

    random.seed(RANDOM_SEED)
    return random.sample(payloads, MAX_REQUESTS)


def fetch_prediction(payload: SalaryPredictionInput) -> dict[str, str | int | float]:
    response = requests.get(API_URL, params=payload.model_dump(), timeout=30)
    response.raise_for_status()

    body = response.json()
    result = payload.model_dump()
    result["predicted_salary_in_usd"] = body["predicted_salary_in_usd"]
    return result


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    payloads = generate_inputs()
    results: list[dict[str, str | int | float]] = []
    failures = 0

    print(f"Starting batch prediction run with {len(payloads)} requests.")

    for index, payload in enumerate(payloads, start=1):
        if index == 1 or index % PROGRESS_EVERY == 0 or index == len(payloads):
            print(f"Processing request {index}/{len(payloads)}...")

        try:
            results.append(fetch_prediction(payload))
        except requests.HTTPError as exc:
            failures += 1
            print(f"[{index}/{len(payloads)}] API error for {payload.model_dump()}: {exc}")
        except requests.RequestException as exc:
            failures += 1
            print(f"[{index}/{len(payloads)}] Request failed: {exc}")

    if not results:
        print("No predictions were collected. Make sure the FastAPI server is running.")
        return

    df = pd.DataFrame(results).sort_values(
        by=["job_title_clean", "experience_level", "company_size", "remote_ratio"]
    )
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(payloads)} request payloads.")
    print(f"Saved {len(results)} predictions to: {OUTPUT_PATH}")
    print(f"Failed requests: {failures}")


if __name__ == "__main__":
    main()
