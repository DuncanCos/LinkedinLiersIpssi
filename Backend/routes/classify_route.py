from fastapi import APIRouter

from controllers.classify_controller import classify_text_all, get_available_models
from schemas.classify_schema import ClassifyRequest, ClassifyResponse, ModelsResponse


router = APIRouter(tags=["classification"])


@router.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(models=get_available_models())


@router.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest) -> ClassifyResponse:
    prediction, predictions = classify_text_all(payload.text)
    return ClassifyResponse(
        prediction=prediction,
        predictions=predictions,
        model_count=len(predictions),
    )
