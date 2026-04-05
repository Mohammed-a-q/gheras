FROM python:3.10-slim

WORKDIR /app

# Copy project files with explicit directories
COPY . .
COPY templates/ templates/
COPY static/ static/

# Make start script executable
RUN chmod +x start.sh

# Install PyTorch CPU-only first (separate to avoid timeout)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: Model download removed from build step.
# Models will be lazily loaded on first request (see get_classifier() in main.py)
# This prevents startup hangs and allows quick healthcheck response.

# Expose port (default value for local testing)
ENV PORT=8000
EXPOSE 8000

# Start the application using the script (ensures $PORT expansion)
CMD ["./start.sh"]
