# Single service: FastAPI (API + live federation server) serving the built React dashboard.
# Build context = the "Multi-Hospital Disease Prediction" folder.
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN node node_modules/vite/bin/vite.js build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=10000 STATE_DIR=/app/state
WORKDIR /app
COPY requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt
COPY ml ./ml
COPY backend ./backend
COPY models ./models
COPY results ./results
COPY data/clients/test.csv ./data/clients/test.csv
COPY --from=web /web/dist ./frontend/dist
RUN useradd --create-home app && mkdir -p /app/state && chown -R app /app/state
USER app
EXPOSE 10000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
