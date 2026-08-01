from pydantic import BaseModel, Field


class CountyFeatures(BaseModel):
    population: int = Field(gt=0)
    population_lag_1: float | None = Field(default=None, gt=0)
    growth_rate_lag_1: float | None = None
    median_household_income: float | None = Field(default=None, gt=0)
    median_age: float | None = Field(default=None, gt=0, lt=120)
    poverty_rate: float | None = Field(default=None, ge=0, le=1)
    unemployment_rate: float | None = Field(default=None, ge=0, le=1)
    housing_vacancy_rate: float | None = Field(default=None, ge=0, le=1)
    region: str = "South"


class PredictionResponse(BaseModel):
    prediction: int
    growth_probability: float
    model_type: str
