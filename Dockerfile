FROM python:3.10-slim

WORKDIR /app

# Copy project files
COPY . .

# Install PyTorch CPU-only first (separate to avoid timeout)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the ML model if possible (non-fatal if fails)
RUN python download_model.py || echo "Model will be downloaded on first request"

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
