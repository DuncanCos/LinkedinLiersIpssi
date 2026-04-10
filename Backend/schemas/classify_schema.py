from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")


class ClassifyResponse(BaseModel):
    prediction: int = Field(..., ge=0, le=1)

