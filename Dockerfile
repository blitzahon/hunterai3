# Stage 1: Build the Vite Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Build Python Backend
FROM python:3.12-slim
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Copy compiled frontend build into static directory
COPY --from=frontend-builder /app/frontend/dist ./static

EXPOSE 5000

CMD ["uvicorn", "api_fastapi:app", "--host", "0.0.0.0", "--port", "5000"]