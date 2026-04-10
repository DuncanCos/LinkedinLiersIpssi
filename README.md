# LinkedIn Liers

Application full-stack pour:

- résumer un texte (via OpenRouter),
- classifier un texte avec plusieurs modèles `.h5` (vote majoritaire),
- afficher le résultat dans une interface React.

## Stack

- Frontend: React + Vite
- Backend: FastAPI
- IA/NLP: TensorFlow, Transformers, PyTorch

## Structure du projet

```text
.
├── Backend/              # API FastAPI + chargement des modèles .h5
├── Front/                # Interface React
├── model/LSTM/           # Notebook/script liés aux modèles
├── MLP.h5                # Exemple de modèle (racine)
└── README.md
```

## Prérequis

- Node.js 18+ (recommandé)
- Python 3.10+ (recommandé)
- `pip` et `npm`

## Configuration des variables d'environnement

Créer `Backend/.env` (ou un `.env` à la racine) avec:

```env
OPENROUTER_API_KEY=...
# Optionnel, pour télécharger le modèle de traduction Hugging Face:
HF_TOKEN=...
```

`OPENROUTER_API_KEY` est obligatoire pour l'endpoint de résumé.

## Installation

### 1) Backend

```bash
cd Backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Frontend

```bash
cd Front
npm install
```

## Lancer le projet en local

### 1) Démarrer l'API

Depuis `Backend/`:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

API disponible sur `http://127.0.0.1:8000`.

### 2) Démarrer le frontend

Depuis `Front/`:

```bash
npm run dev
```

Frontend disponible sur `http://127.0.0.1:5173`.

Le proxy Vite redirige `/api/*` vers `http://127.0.0.1:8000/*`.

## Endpoints API

- `POST /summaries`
  - body:
    ```json
    { "text": "..." }
    ```
  - response:
    ```json
    { "summary": "..." }
    ```

- `GET /models`
  - response:
    ```json
    { "models": ["LSTM2", "MLP", "..."] }
    ```

- `POST /classify`
  - body:
    ```json
    { "text": "..." }
    ```
  - response:
    ```json
    {
      "prediction": 1,
      "predictions": { "LSTM2": 1, "MLP": 0 },
      "model_count": 2
    }
    ```

## Notes techniques

- Les modèles de classification sont chargés automatiquement depuis les fichiers `Backend/*.h5`.
- Le backend tente une traduction FR -> EN avant classification (fallback sur texte source si échec).
- Le frontend appelle en parallèle `/summaries` et `/classify`.
