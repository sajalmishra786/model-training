import requests

url = "http://127.0.0.1:8000/predict"


data = {
    "features": [0] * 52
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response Text:", response.text)