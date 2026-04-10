from fastapi import APIRouter

from controllers.summarize_controller import generate_summary
from schemas.summarize_schema import SummarizeRequest, SummarizeResponse


router = APIRouter(tags=["summaries"])


@router.post("/summaries", response_model=SummarizeResponse)
def create_summary(payload: SummarizeRequest) -> SummarizeResponse:
    summary = generate_summary(payload.text)
    return SummarizeResponse(summary=summary)
