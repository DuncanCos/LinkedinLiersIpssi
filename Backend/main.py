from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.classify_route import router as classify_router
from routes.summarize_route import router as summarize_router


app = FastAPI(title="LinkedIn Liers API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(summarize_router)
app.include_router(classify_router)
