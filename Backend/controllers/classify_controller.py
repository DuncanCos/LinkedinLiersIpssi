from functools import lru_cache
import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException


TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-fr-en"
MODELS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_NAME = "LSTM2"


def _read_token_from_dotenv() -> str | None:
    env_candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]

    for env_path in env_candidates:
        if not env_path.exists():
            continue
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "HF_TOKEN":
                    parsed = value.strip().strip('"').strip("'")
                    if parsed:
                        return parsed
        except OSError:
            continue
    return None


def _get_hf_token() -> str | None:
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token
    return _read_token_from_dotenv()


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
def _get_translation_components() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Transformers is not installed. "
                "Install it to enable translation before classification."
            ),
        ) from exc

    hf_token = _get_hf_token()
    auth_kwargs: dict[str, str] = {}
    if hf_token:
        auth_kwargs["token"] = hf_token

    try:
        tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME, **auth_kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME, **auth_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load translation model: {exc}",
        ) from exc

    return tokenizer, model


def _translate_to_english(text: str) -> str:
    if not text.strip():
        return text

    try:
        tokenizer, translation_model = _get_translation_components()
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        generated = translation_model.generate(
            **encoded,
            max_length=512,
        )
        translated = tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        ).strip()
        return translated if translated else text
    except Exception:  # noqa: BLE001
        # Translation is an optional pre-processing step: fallback to source text.
        return text


@lru_cache(maxsize=1)
def _get_model_registry() -> dict[str, tuple[str, Path]]:
    model_files = sorted(MODELS_DIR.glob("*.h5"), key=lambda path: path.stem.lower())
    if not model_files:
        raise HTTPException(
            status_code=500,
            detail=f"No .h5 model found in {MODELS_DIR}",
        )

    registry: dict[str, tuple[str, Path]] = {}
    for model_path in model_files:
        model_name = model_path.stem
        registry[model_name.upper()] = (model_name, model_path)
    return registry


def get_available_models() -> list[str]:
    registry = _get_model_registry()
    return sorted((entry[0] for entry in registry.values()), key=str.lower)


def _normalize_model_name(model_name: str | None) -> str:
    registry = _get_model_registry()

    requested = (model_name or "").strip()
    if not requested:
        fallback_key = DEFAULT_MODEL_NAME.upper()
        if fallback_key in registry:
            return registry[fallback_key][0]
        return next(iter(sorted((entry[0] for entry in registry.values()), key=str.lower)))

    normalized = requested.upper()
    if normalized not in registry:
        supported = ", ".join(get_available_models())
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{model_name}'. Supported models: {supported}.",
        )
    return registry[normalized][0]


@lru_cache(maxsize=16)
def _get_model(model_name: str) -> Any:
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

    registry = _get_model_registry()
    model_entry = registry.get(model_name.upper())
    if model_entry is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{model_name}'.",
        )

    _, model_path = model_entry

    if not model_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {model_path}",
        )

    try:
        return keras.models.load_model(model_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {exc}",
        ) from exc


def _predict_with_model(text: str, model_name: str) -> int:
    model = _get_model(model_name)
    model_input = _prepare_text_input(model, text)
    try:
        raw_prediction = model.predict(model_input, verbose=0)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run prediction with model '{model_name}': {exc}",
        ) from exc

    score = _extract_score(raw_prediction)
    return 1 if score >= 0.5 else 0


def classify_text(text: str, model_name: str | None = None) -> tuple[int, str]:
    normalized_model_name = _normalize_model_name(model_name)
    translated_text = _translate_to_english(text)
    prediction = _predict_with_model(translated_text, normalized_model_name)
    return prediction, normalized_model_name


def classify_text_all(text: str) -> tuple[int, dict[str, int]]:
    translated_text = _translate_to_english(text)
    model_names = get_available_models()

    predictions: dict[str, int] = {}
    for model_name in model_names:
        predictions[model_name] = _predict_with_model(translated_text, model_name)

    positive_votes = sum(predictions.values())
    majority = 1 if positive_votes >= (len(predictions) / 2) else 0
    return majority, predictions
