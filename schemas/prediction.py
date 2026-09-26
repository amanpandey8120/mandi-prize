"""
Pydantic schemas for the /predict endpoint.
"""

from pydantic import BaseModel, field_validator


class PredictionRequest(BaseModel):
    """Request body for POST /predict."""

    state: str
    district: str
    commodity: str

    @field_validator("state", "district", "commodity", mode="before")
    @classmethod
    def strip_and_non_empty(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Must be a string")
        v = v.strip()
        if not v:
            raise ValueError("Must not be empty")
        return v


class PredictionResponse(BaseModel):
    """Response body for POST /predict."""

    state: str
    district: str
    commodity: str
    predicted_price: float


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    model_loaded: bool
