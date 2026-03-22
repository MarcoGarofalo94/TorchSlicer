# ── Stage 1: build React app ──────────────────────────────────────────────────
FROM node:20-alpine AS build
WORKDIR /build

# Install deps first (cached unless package.json changes)
COPY frontend/package.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ── Stage 2: FastAPI backend ───────────────────────────────────────────────────
FROM python:3.12-slim
RUN pip install --no-cache-dir fastapi "uvicorn[standard]==0.24.0" requests

WORKDIR /app
COPY app.py .
COPY --from=build /build/dist ./static

EXPOSE 8080
CMD ["python", "app.py"]
