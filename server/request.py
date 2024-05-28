import requests
import json

url = "http://127.0.0.1:8000/upload"

files = {
    "file" : open("prafull.jpg", "rb")
}
data = {
    "model_num": "2"
}

response = requests.post(url, files=files, data={'data': json.dumps(data)})

print(response.json())