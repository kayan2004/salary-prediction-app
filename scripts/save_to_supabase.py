from __future__ import annotations

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import OLLAMA_MODEL
from services.supabase import persist_run

PREDICTIONS_PATH = BASE_DIR / "data" / "predictions.csv"
ANALYSIS_PATH = BASE_DIR / "data" / "analysis.txt"
CHART_PATH = BASE_DIR / "data" / "charts" / "salary_by_role.png"


def main() -> None:
    run_id, prediction_count = persist_run(
        predictions_path=PREDICTIONS_PATH,
        analysis_path=ANALYSIS_PATH,
        chart_path=CHART_PATH,
        model_name=OLLAMA_MODEL,
    )

    print(f"Saved run {run_id} to Supabase.")
    print(f"Saved {prediction_count} predictions.")


if __name__ == "__main__":
    main()
