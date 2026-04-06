from fastapi.testclient import TestClient
import main
from PIL import Image
import io

client = TestClient(main.app)
print('GET / status', client.get('/').status_code)
img = Image.new('RGB', (100, 100), (0, 200, 0))
buf = io.BytesIO()
img.save(buf, format='PNG')
buf.seek(0)
res = client.post('/analyze', files={'file': ('test.png', buf, 'image/png')}, data={'city': 'Riyadh'})
print('POST /analyze status', res.status_code)
print(res.text[:800])
