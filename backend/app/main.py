
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.documents import router as documents_router
from backend.app.services.indiclid_service import IndicLIDService


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


# Allow our React frontend to communicate with the backend.
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


# Register document API routes.
app.include_router(
    documents_router,
    prefix="/documents",
    tags=["Documents"],
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
