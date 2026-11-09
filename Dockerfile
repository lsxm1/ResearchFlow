# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim AS web
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RF_DATA_DIR=/data RF_TESSDATA_DIR=/tessdata
COPY backend/requirements-lock.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --uid 10001 --no-create-home researchflow \
    && mkdir -p /data /tessdata \
    && chown researchflow /data
ADD --chmod=0644 --checksum=sha256:7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2 https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/eng.traineddata /tessdata/eng.traineddata
COPY backend/app/ ./backend/app/
COPY --from=web /build/dist ./frontend/dist/
USER researchflow
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
