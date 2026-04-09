# Salary Prediction Application

End-to-end machine learning application for predicting data science salaries, generating business-facing insights, storing results in Supabase, and presenting them in a Streamlit dashboard.

## Project Overview

This project uses the Kaggle **Data Science Job Salaries** dataset to train a `DecisionTreeRegressor` that predicts salary in USD based on role and job attributes such as:

- experience level
- employment type
- company size
- job title
- employee residence
- company location
- remote ratio

The application then:

- serves predictions through a FastAPI API
- generates a summary and charts from batch prediction results
- stores runs and prediction records in Supabase
- displays the latest run in a Streamlit dashboard

## Architecture

`Training Script -> FastAPI Prediction API -> Batch Prediction Script -> Analysis + Charts -> Supabase -> Streamlit Dashboard`

## Main Components

- `scripts/train_model.py`
  Retrains the salary prediction model and saves the model bundle and training metrics.

- `main.py` + `routers/salary.py`
  Exposes the salary prediction API through FastAPI.

- `scripts/api_call.py`
  Generates valid input combinations, calls the API, and saves predictions to `data/predictions.csv`.

- `scripts/analyze_predictions.py`
  Builds business-facing analysis text and generates charts from prediction output.

- `scripts/save_to_supabase.py`
  Persists the latest run metadata and prediction rows to Supabase.

- `dashboard/app.py`
  Displays the latest run, metrics, tables, and charts using Streamlit.

## Tech Stack

- Python
- FastAPI
- Streamlit
- scikit-learn
- pandas
- matplotlib
- Supabase
- Ollama

## Run Locally

1. Create and activate the virtual environment.
2. Install dependencies from `requirements.txt`.
3. Add a `.env` file in the project root with:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key
OLLAMA_MODEL=your_ollama_model
```

4. Retrain the model if needed:

```powershell
.\.venv\Scripts\python scripts\train_model.py
```

5. Start the FastAPI app:

```powershell
.\.venv\Scripts\uvicorn main:app --reload
```

6. Generate predictions:

```powershell
.\.venv\Scripts\python scripts\api_call.py
```

7. Generate analysis and charts:

```powershell
.\.venv\Scripts\python scripts\analyze_predictions.py
```

8. Save results to Supabase:

```powershell
.\.venv\Scripts\python scripts\save_to_supabase.py
```

9. Launch the dashboard:

```powershell
.\.venv\Scripts\python -m streamlit run dashboard/app.py
```

## Deliverables

- Live Streamlit App URL: https://salary-prediction-app-production.up.railway.app/
- Deployed FastAPI Endpoint URL: https://fastapi-endpoint-production-412a.up.railway.app
- README: included in this repository


