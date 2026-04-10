# Frontend (Vite + React)

Le frontend est connecte au backend FastAPI via:

- un proxy Vite `"/api" -> "http://127.0.0.1:8000"`
- un appel frontend sur `POST /api/summaries`
- un appel frontend sur `POST /api/classify`

## Lancer le projet

1. Lancer le backend (depuis le dossier `Backend`) sur `http://127.0.0.1:8000`.
2. Lancer le frontend:

```bash
cd Front
npm install
npm run dev
```

Le frontend est servi par defaut sur `http://127.0.0.1:5173`.

## Option: URL API custom

Par defaut, le frontend utilise `VITE_API_BASE_URL=/api`.
Tu peux la surcharger dans un fichier `.env` du front:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```
