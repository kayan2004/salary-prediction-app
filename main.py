from fastapi import FastAPI

from routers.chat import router as chat_router
from routers.salary import router as salary_router

app = FastAPI(
    title="Salary Prediction API",
    description="Predict salaries for data science roles using a trained decision tree model.",
    version="1.0.0",
)

app.include_router(salary_router)
app.include_router(chat_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
