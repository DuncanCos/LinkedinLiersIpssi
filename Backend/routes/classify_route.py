from fastapi import APIRouter

from controllers.classify_controller import classify_text
from schemas.classify_schema import ClassifyRequest, ClassifyResponse


router = APIRouter(tags=["classification"])


@router.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest) -> ClassifyResponse:
    prediction = classify_text(payload.text)
    return ClassifyResponse(prediction=prediction)

