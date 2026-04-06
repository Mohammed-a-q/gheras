#!/bin/bash
# Run the FastAPI app with Uvicorn on port 8000
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --no-access-log --log-level warning
