from fastapi import FastAPI

from routes.summarize_route import router as summarize_router


app = FastAPI(title="LinkedIn Liers API")
app.include_router(summarize_router)
