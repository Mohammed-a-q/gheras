FROM python:3.10-slim

WORKDIR /app

# Copy project files
COPY . .

# Install PyTorch CPU-only first (separate to avoid timeout)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: Model download removed from build step.
# Models will be lazily loaded on first request (see get_classifier() in main.py)
# This prevents startup hangs and allows quick healthcheck response.

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
