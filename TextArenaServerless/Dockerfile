# Use official Python slim image
FROM python:3.12.5-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy all files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Expose port used by FastAPI/Uvicorn
EXPOSE 8000

# Environment variables (can be overridden when running the container)
ENV SUPABASE_URL=...
ENV SUPABASE_KEY=...
ENV WEBSOCKET_PORT=8000

# Start the FastAPI server using main.py
CMD ["python", "main.py"]
