import requests
from PIL import Image
import numpy as np
import io

img = Image.fromarray(np.random.randint(0, 255, (800, 1000, 3), dtype=np.uint8))
buf = io.BytesIO()
img.save(buf, format='PNG')
buf.seek(0)

try:
    resp = requests.post('http://127.0.0.1:5000/api/diagnose/composite', files={'composite': ('test.png', buf, 'image/png')})
    print('STATUS CODE:', resp.status_code)
    print(resp.json())
except Exception as e:
    print('ERROR:', e)
