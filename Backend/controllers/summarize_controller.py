import os

import requests
from fastapi import HTTPException

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemma-3-4b-it"


def generate_summary(text: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Missing OPENROUTER_API_KEY environment variable.",
        )

    prompt = (
        "Make a short summary in exactly 1 sentences of this text:\n\n"
        f"{text}\n\n"
        "Return only the summary, respond in french."
    )

    try:
        response = requests.post(
            url=OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "reasoning": {"enabled": True},
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach OpenRouter: {exc}",
        ) from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter error: {response.status_code} - {response.text}",
        )

    try:
        payload = response.json()
        summary = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Invalid response format from OpenRouter.",
        ) from exc

    if not isinstance(summary, str) or not summary.strip():
        raise HTTPException(
            status_code=502,
            detail="OpenRouter returned an empty summary.",
        )

    return summary.strip()
