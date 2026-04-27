FROM python:3.11-slim-bookworm

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        supervisor && \
    rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────
WORKDIR /app

# ── Install Python deps first (layer cache) ─────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ──────────────────────────────────────
COPY . .

# ── Supervisor config (runs both uvicorn + streamlit) ────────
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# ── Expose ports ─────────────────────────────────────────────
#    8000  → FastAPI  (uvicorn)
#    8501  → Streamlit
EXPOSE 8000 8501

# ── Health-check on FastAPI ──────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Launch both services via supervisor ──────────────────────
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
