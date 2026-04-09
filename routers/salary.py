from fastapi import APIRouter, Depends

from schemas.salary import SalaryPredictionInput, SalaryPredictionResponse
from services.model import predict_salary

router = APIRouter(tags=["salary"])


@router.get("/predict_salary", response_model=SalaryPredictionResponse)
def predict_salary_route(
    params: SalaryPredictionInput = Depends(),
) -> SalaryPredictionResponse:
    predicted_salary = predict_salary(params)
    return SalaryPredictionResponse(predicted_salary_in_usd=predicted_salary)
