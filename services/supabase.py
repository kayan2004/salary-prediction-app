from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import SUPABASE_KEY, SUPABASE_URL


class SupabaseClient:
    def __init__(self) -> None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set.")

        self.base_url = f"{SUPABASE_URL.rstrip('/')}/rest/v1"
        self.headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    def insert_prediction_run(
        self,
        model_name: str,
        analysis_text: str | None,
        chart_path: str | None,
    ) -> int:
        payload = {
            "model_name": model_name,
            "analysis_text": analysis_text,
            "chart_path": chart_path,
        }
        response = requests.post(
            f"{self.base_url}/prediction_runs",
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data[0]["id"]

    def insert_predictions(self, run_id: int, df: pd.DataFrame) -> int:
        rows = df.copy()
        rows["run_id"] = run_id

        payload = rows.to_dict(orient="records")
        response = requests.post(
            f"{self.base_url}/predictions",
            headers=self.headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return len(response.json())


def load_predictions_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found at {path}")
    return pd.read_csv(path)


def load_text_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def persist_run(
    predictions_path: Path,
    analysis_path: Path,
    chart_path: Path,
    model_name: str,
) -> tuple[int, int]:
    client = SupabaseClient()
    predictions_df = load_predictions_csv(predictions_path)
    analysis_text = load_text_file(analysis_path)

    saved_run_id = client.insert_prediction_run(
        model_name=model_name,
        analysis_text=analysis_text,
        chart_path=str(chart_path) if chart_path.exists() else None,
    )
    saved_prediction_count = client.insert_predictions(saved_run_id, predictions_df)
    return saved_run_id, saved_prediction_count
