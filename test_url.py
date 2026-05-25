import requests
import json

url = "http://127.0.0.1:8000/analyze-url"
payload = {
    "url": "https://www.kozco.com/tech/piano2.wav",
    "params": {
        "rpm": 3000,
        "n": 9,
        "d": 7.5,
        "D": 39,
        "alpha": 0
    }
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    data = response.json()
    if data.get("success"):
        print("Success! Fault Freqs:", data.get("fault_freqs"))
        print("Indicators:", data.get("indicators"))
    else:
        print("Error:", data.get("error"))
except Exception as e:
    print("Connection error:", e)
