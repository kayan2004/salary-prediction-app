import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Salary Intelligence Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_role_label(label: str) -> str:
    if label == "Other":
        return "Grouped specialist roles"
    return label


def get_latest_run() -> dict | None:
    url = (
        f"{SUPABASE_URL}/rest/v1/prediction_runs"
        "?select=*&order=created_at.desc&limit=1"
    )
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    rows = response.json()
    return rows[0] if rows else None


def get_predictions_for_run(run_id: int) -> list[dict]:
    url = f"{SUPABASE_URL}/rest/v1/predictions?select=*&run_id=eq.{run_id}"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


load_dotenv_file(ENV_PATH)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials are missing. Check your .env file.")
    st.stop()

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
}

latest_run = get_latest_run()
if latest_run is None:
    st.warning("No prediction runs found in Supabase yet.")
    st.stop()

predictions = get_predictions_for_run(latest_run["id"])
if not predictions:
    st.warning("No predictions found for the latest run.")
    st.stop()

predictions_df = pd.DataFrame(predictions)
predictions_df["predicted_salary_in_usd"] = predictions_df["predicted_salary_in_usd"].astype(float)
predictions_df["job_title_display"] = predictions_df["job_title_clean"].map(format_role_label)

role_df = (
    predictions_df.groupby("job_title_display", as_index=False)["predicted_salary_in_usd"]
    .mean()
    .sort_values("predicted_salary_in_usd", ascending=False)
)
experience_df = (
    predictions_df.groupby("experience_level", as_index=False)["predicted_salary_in_usd"]
    .mean()
    .sort_values("predicted_salary_in_usd", ascending=False)
)
remote_df = (
    predictions_df.groupby("remote_ratio", as_index=False)["predicted_salary_in_usd"]
    .mean()
    .sort_values("remote_ratio")
)

avg_salary = predictions_df["predicted_salary_in_usd"].mean()
highest_role = role_df.iloc[0]
lowest_role = role_df.iloc[-1]
top_experience = experience_df.iloc[0]

st.title("Salary Intelligence Dashboard")
st.caption("Latest model run from Supabase, presented for quick business review.")

with st.container(border=True):
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        st.subheader("Executive Summary")
        analysis_text = latest_run.get("analysis_text")
        if analysis_text:
            st.write(analysis_text)
        else:
            st.info("No analysis text found for this run.")
    with right:
        st.subheader("Run Details")
        st.write(f"Run ID: `{latest_run['id']}`")
        st.write(f"Created at: `{latest_run['created_at']}`")
        st.write(f"Model: `{latest_run['model_name']}`")
        st.write(f"Scenarios analyzed: `{len(predictions_df):,}`")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Average Salary", format_currency(avg_salary))
metric_2.metric("Highest Role", highest_role["job_title_display"], format_currency(highest_role["predicted_salary_in_usd"]))
metric_3.metric("Lowest Role", lowest_role["job_title_display"], format_currency(lowest_role["predicted_salary_in_usd"]))
metric_4.metric("Top Experience Tier", top_experience["experience_level"], format_currency(top_experience["predicted_salary_in_usd"]))

chart_col, table_col = st.columns([1.4, 1], gap="large")

with chart_col:
    st.subheader("Average Predicted Salary by Role")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(role_df["job_title_display"], role_df["predicted_salary_in_usd"], color="#1f77b4")
    ax.set_ylabel("Average Salary (USD)")
    ax.set_xlabel("Job Title")
    ax.set_title("Role Comparison")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

with table_col:
    st.subheader("Snapshot")
    st.write(f"Highest-paying role: **{highest_role['job_title_display']}**")
    st.write(f"Average salary for that role: **{format_currency(highest_role['predicted_salary_in_usd'])}**")
    st.write(f"Lowest-paying role: **{lowest_role['job_title_display']}**")
    st.write(f"Average salary for that role: **{format_currency(lowest_role['predicted_salary_in_usd'])}**")
    st.write(f"Top experience tier: **{top_experience['experience_level']}**")
    st.write(f"Average salary for that tier: **{format_currency(top_experience['predicted_salary_in_usd'])}**")

extra_chart_col_1, extra_chart_col_2 = st.columns(2, gap="large")

with extra_chart_col_1:
    st.subheader("Salary by Experience Level")
    ordered_experience = ["EN", "MI", "SE", "EX"]
    experience_chart_df = experience_df.copy()
    experience_chart_df["experience_level"] = pd.Categorical(
        experience_chart_df["experience_level"],
        categories=ordered_experience,
        ordered=True,
    )
    experience_chart_df = experience_chart_df.sort_values("experience_level")

    fig_exp, ax_exp = plt.subplots(figsize=(7, 4.5))
    ax_exp.plot(
        experience_chart_df["experience_level"],
        experience_chart_df["predicted_salary_in_usd"],
        marker="o",
        linewidth=3,
        color="#2ca02c",
    )
    ax_exp.set_title("Compensation Growth Across Experience Tiers")
    ax_exp.set_xlabel("Experience Level")
    ax_exp.set_ylabel("Average Salary (USD)")
    ax_exp.spines["top"].set_visible(False)
    ax_exp.spines["right"].set_visible(False)
    ax_exp.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig_exp, use_container_width=True)

with extra_chart_col_2:
    st.subheader("Salary by Remote Ratio")
    remote_chart_df = remote_df.copy()
    remote_chart_df["remote_label"] = remote_chart_df["remote_ratio"].map(
        {
            0: "On-site",
            50: "Hybrid",
            100: "Remote",
        }
    )

    fig_remote, ax_remote = plt.subplots(figsize=(7, 4.5))
    bars = ax_remote.bar(
        remote_chart_df["remote_label"],
        remote_chart_df["predicted_salary_in_usd"],
        color=["#264653", "#e9c46a", "#e76f51"],
    )
    ax_remote.set_title("Work Setup and Salary Outlook")
    ax_remote.set_xlabel("Work Setup")
    ax_remote.set_ylabel("Average Salary (USD)")
    ax_remote.spines["top"].set_visible(False)
    ax_remote.spines["right"].set_visible(False)
    ax_remote.grid(axis="y", linestyle="--", alpha=0.25)
    for bar in bars:
        height = bar.get_height()
        ax_remote.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            format_currency(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    st.pyplot(fig_remote, use_container_width=True)

st.subheader("Prediction Records")
display_df = predictions_df.copy()
display_df["job_title_clean"] = display_df["job_title_display"]
display_df = display_df.drop(columns=["job_title_display"])
display_df["predicted_salary_in_usd"] = display_df["predicted_salary_in_usd"].map(format_currency)
st.dataframe(display_df, use_container_width=True, hide_index=True)
