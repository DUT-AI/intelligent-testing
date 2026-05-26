from fastapi import FastAPI
import uvicorn
from dishka import make_container
from dishka.integrations.fastapi import setup_dishka

from app.infrastructure.api.routers import router as user_router
from app.infrastructure.di import AppProvider

# Initialize the main FastAPI application
app = FastAPI(
    title="Intelligent Testing Core API",
    description="Clean Architecture implementation with FastAPI, SQLAlchemy & Dishka DI",
    version="1.0.0",
)

# Setup Dishka container and integrate with FastAPI
container = make_container(AppProvider())
setup_dishka(container, app)

# Include routers
app.include_router(user_router)


@app.get("/health")
def health_check():
    return {
        "status": "online",
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
