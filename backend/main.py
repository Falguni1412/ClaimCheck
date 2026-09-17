"""
ClaimCheck backend entrypoint.

Run from the `backend` directory:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Or:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
from api.main import app

__all__ = ["app"]
