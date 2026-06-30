"""
Vercel serverless entry point for the FastAPI backend.
This file imports the FastAPI app from main.py and exposes it
for Vercel's Python runtime.
"""

import sys
import os

# Ensure the project root is on the Python path so we can import main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402, F401
