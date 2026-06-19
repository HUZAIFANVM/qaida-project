"""Start the FastAPI server."""
import uvicorn

uvicorn.run(
    "qaida_project.api.main:app",
    host="0.0.0.0",
    port=8000,
    workers=1,  # single worker for GPU model safety
)
