from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from schemas.chat import ChatMessage
from services.ollama import chat_completion

PREDICTIONS_PATH = BASE_DIR / "data" / "predictions.csv"
OUTPUT_PATH = BASE_DIR / "data" / "analysis.txt"
CHARTS_DIR = BASE_DIR / "data" / "charts"
CHART_PATH = CHARTS_DIR / "salary_by_role.png"


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Predictions file not found at {PREDICTIONS_PATH}")
    return pd.read_csv(PREDICTIONS_PATH)


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_role_label(label: str) -> str:
    if label == "Other":
        return "Grouped specialist roles"
    return label


def build_findings(df: pd.DataFrame) -> dict[str, object]:
    role_df = (
        df.groupby("job_title_clean", as_index=False)["predicted_salary_in_usd"]
        .mean()
        .sort_values("predicted_salary_in_usd", ascending=False)
    )
    experience_df = (
        df.groupby("experience_level", as_index=False)["predicted_salary_in_usd"]
        .mean()
        .sort_values("predicted_salary_in_usd", ascending=False)
    )
    remote_df = (
        df.groupby("remote_ratio", as_index=False)["predicted_salary_in_usd"]
        .mean()
        .sort_values("predicted_salary_in_usd", ascending=False)
    )
    combo_df = (
        df.groupby(["job_title_clean", "experience_level"], as_index=False)["predicted_salary_in_usd"]
        .mean()
        .sort_values("predicted_salary_in_usd", ascending=False)
    )

    top_role = role_df.iloc[0]
    bottom_role = role_df.iloc[-1]
    top_experience = experience_df.iloc[0]
    entry_level = experience_df.loc[experience_df["experience_level"] == "EN"].iloc[0]
    remote_best = remote_df.iloc[0]
    top_combo = combo_df.iloc[0]

    remote_label = {
        0: "on-site",
        50: "hybrid",
        100: "remote",
    }

    return {
        "scenario_count": int(len(df)),
        "overall_average_salary": float(df["predicted_salary_in_usd"].mean()),
        "top_role": format_role_label(str(top_role["job_title_clean"])),
        "top_role_salary": float(top_role["predicted_salary_in_usd"]),
        "bottom_role": format_role_label(str(bottom_role["job_title_clean"])),
        "bottom_role_salary": float(bottom_role["predicted_salary_in_usd"]),
        "top_experience": str(top_experience["experience_level"]),
        "top_experience_salary": float(top_experience["predicted_salary_in_usd"]),
        "entry_level_salary": float(entry_level["predicted_salary_in_usd"]),
        "best_remote_setup": remote_label[int(remote_best["remote_ratio"])],
        "best_remote_salary": float(remote_best["predicted_salary_in_usd"]),
        "top_combo_role": format_role_label(str(top_combo["job_title_clean"])),
        "top_combo_experience": str(top_combo["experience_level"]),
        "top_combo_salary": float(top_combo["predicted_salary_in_usd"]),
    }


def build_llm_prompt(findings: dict[str, object]) -> str:
    return f"""
You are a business analyst.

Use only these findings from a salary prediction run:
- Scenario count: {findings['scenario_count']}
- Overall average salary: {format_currency(findings['overall_average_salary'])}
- Highest-paying role: {findings['top_role']} at {format_currency(findings['top_role_salary'])}
- Lowest-paying role: {findings['bottom_role']} at {format_currency(findings['bottom_role_salary'])}
- Highest-paying experience tier: {findings['top_experience']} at {format_currency(findings['top_experience_salary'])}
- Entry-level average salary: {format_currency(findings['entry_level_salary'])}
- Best work setup in this run: {findings['best_remote_setup']} at {format_currency(findings['best_remote_salary'])}
- Richest role and experience combination: {findings['top_combo_role']} at {findings['top_combo_experience']} reaching {format_currency(findings['top_combo_salary'])}

Write one short paragraph that summarizes the run naturally.
Keep it under 100 words.
Mention the main role trend, the experience-level pattern, and the work-setup result.
Include one cautious business implication stated descriptively, not as advice.
Write naturally and avoid awkward phrasing.
Do not repeat labels mechanically.
Use only the facts above.
Do not invent numbers or explanations.
Do not recommend actions.
Do not generalize beyond the stated facts.
Do not use bullet points.
""".strip()


def output_is_usable(text: str) -> bool:
    lowered = text.lower()
    banned_signals = [
        "as an ai",
        "our services",
        "platform",
        "valued clients",
        "business challenges",
        "comprehensive suite",
    ]
    if any(signal in lowered for signal in banned_signals):
        return False
    if len(text.split()) > 90:
        return False
    return True


def rewrite_summary_with_llm(findings: dict[str, object]) -> str:
    prompt = build_llm_prompt(findings)
    try:
        _, content = asyncio.run(
            chat_completion(
                messages=[
                    ChatMessage(role="system", content="You summarize structured findings clearly and naturally without adding unsupported claims."),
                    ChatMessage(role="user", content=prompt),
                ]
            )
        )
    except Exception:
        return "Analysis could not be generated."

    cleaned = content.strip()
    if not cleaned or not output_is_usable(cleaned):
        return "Analysis could not be generated."
    return cleaned


def create_chart(df: pd.DataFrame) -> Path:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    chart_df = (
        df.groupby("job_title_clean", as_index=False)["predicted_salary_in_usd"]
        .mean()
        .sort_values("predicted_salary_in_usd", ascending=False)
    )
    chart_df["job_title_clean"] = chart_df["job_title_clean"].map(format_role_label)

    plt.figure(figsize=(12, 6))
    plt.bar(chart_df["job_title_clean"], chart_df["predicted_salary_in_usd"], color="#2a6f97")
    plt.title("Average Predicted Salary by Job Title")
    plt.xlabel("Job Title")
    plt.ylabel("Average Predicted Salary in USD")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.close()

    return CHART_PATH


def main() -> None:
    df = load_predictions()
    chart_path = create_chart(df)
    findings = build_findings(df)
    analysis = rewrite_summary_with_llm(findings)

    OUTPUT_PATH.write_text(analysis, encoding="utf-8")

    print("Analysis saved to:", OUTPUT_PATH)
    print("Chart saved to:", chart_path)
    print()
    print(analysis)


if __name__ == "__main__":
    main()
