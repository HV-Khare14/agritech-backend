"""
Vercel Serverless Entry Point
Exposes the FastAPI app for Vercel's Python runtime.
"""
import sys
from pathlib import Path

# Add the Backend root to sys.path so 'app' package is importable
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.main import app

# Vercel picks up 'app' automatically as the ASGI handler
