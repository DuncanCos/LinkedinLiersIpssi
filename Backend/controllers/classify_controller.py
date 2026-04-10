from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException


MODEL_PATH = Path(__file__).resolve().parent.parent / "LSTM2.h5"


def _extract_score(raw_prediction: Any) -> float:
    current = raw_prediction
    if hasattr(current, "ravel"):
        flattened = current.ravel()
        if getattr(flattened, "size", 0) > 0:
            return float(flattened[0])
    while isinstance(current, (list, tuple)) and current:
        current = current[0]
    try:
        return float(current)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Unexpected model output format.",
        ) from exc


def _prepare_text_input(model: Any, text: str) -> Any:
    model_inputs = getattr(model, "inputs", None)
    if not model_inputs:
        return [text]

    first_input = model_inputs[0]
    input_dtype = getattr(getattr(first_input, "dtype", None), "name", "")
    input_shape = getattr(first_input, "shape", None)

    # If the model embeds text preprocessing layers, it accepts raw strings directly.
    if "string" in str(input_dtype):
        return [text]

    # Numeric input models need token ids. Use hashing + padding as default
    # backend preprocessing when no external tokenizer artifact is provided.
    sequence_length = 100
    if input_shape is not None and len(input_shape) >= 2 and input_shape[1] is not None:
        sequence_length = int(input_shape[1])

    vocab_size = 10000
    embedding_layer = None
    for layer in getattr(model, "layers", []):
        if hasattr(layer, "input_dim"):
            embedding_layer = layer
            break
    if embedding_layer is not None:
        vocab_size = int(getattr(embedding_layer, "input_dim", vocab_size))

    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import one_hot
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="TensorFlow preprocessing utilities are unavailable.",
        ) from exc

    encoded = one_hot(text, n=vocab_size)
    padded = pad_sequences(
        [encoded],
        maxlen=sequence_length,
        padding="post",
        truncating="post",
        dtype="float32",
    )
    return padded


@lru_cache(maxsize=1)
def _get_model() -> Any:
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "TensorFlow is not installed. "
                "Install it to enable the LSTM classifier."
            ),
        ) from exc

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {MODEL_PATH}",
        )

    try:
        return keras.models.load_model(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {exc}",
        ) from exc


def classify_text(text: str) -> int:
    model = _get_model()
    model_input = _prepare_text_input(model, text)

    try:
        raw_prediction = model.predict(model_input, verbose=0)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run prediction: {exc}",
        ) from exc

    score = _extract_score(raw_prediction)
    return 1 if score >= 0.5 else 0
