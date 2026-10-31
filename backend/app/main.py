from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.documents import router as documents_router

# Create our FastAPI application
app = FastAPI(
    title="NitiSaathi API",
    description="Multilingual Government Policy Explainer",
    version="0.1.0",
)

# Allow our React frontend to communicate with the backend
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

app.include_router(documents_router, prefix="/documents", tags=["Documents"])   

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