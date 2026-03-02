"""
Simple test client to POST an image to the local /analyze endpoint.
Usage:
  .\venv\Scripts\python.exe test_upload.py C:\path\to\grass.jpg Riyadh
Requires: requests (pip install requests)
"""
import sys
from pathlib import Path
import requests

def main():
    if len(sys.argv) < 3:
        print("Usage: test_upload.py <image_path> <city>")
        return
    image_path = Path(sys.argv[1])
    city = sys.argv[2]
    if not image_path.exists():
        print(f"File not found: {image_path}")
        return
    url = "http://127.0.0.1:8000/analyze"
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        data = {"city": city}
        print(f"Posting {image_path} -> {url} (city={city})")
        r = requests.post(url, files=files, data=data, timeout=120)
    print("Status:", r.status_code)
    print(r.text[:2000])

if __name__ == '__main__':
    main()
