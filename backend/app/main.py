
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.documents import router as documents_router
from backend.app.api.research import router as research_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize shared machine-learning resources at startup
    and release application references at shutdown.
    """

    print("\n--- Starting NitiSaathi ---")
    # Load the large language detector only when an upload needs it.
    app.state.indiclid_service = None
    print("NitiSaathi is ready to accept requests.\n")

    try:
        yield

    finally:
        # Remove the application's reference to the models.
        app.state.indiclid_service = None

        print("\nIndicLID service references released.")
        print("NitiSaathi shutting down.")


# Create our FastAPI application.
app = FastAPI(
    title="NitiSaathi API",
    description="Multilingual Government Policy Explainer",
    version="0.1.0",
    lifespan=lifespan,
)


# Allow local development and the deployed React frontend to call the API.
# FRONTEND_URL may contain a comma-separated list of deployed frontend origins.
frontend_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "FRONTEND_URL",
        "https://nitisaathi-frontend.onrender.com",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *frontend_origins,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register document API routes.
app.include_router(
    documents_router,
    prefix="/documents",
    tags=["Documents"],
)


# Register the isolated web-research API.
app.include_router(
    research_router,
    prefix="/research",
    tags=["Web Research"],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to NitiSaathi!",
        "description": "Understand government policies in your language.",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "NitiSaathi",
    }
