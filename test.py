import requests
import sys
import json

url = 'http://127.0.0.1:8000/analyze'
try:
    files = {'file': open('audio/1.mp3', 'rb')}
except FileNotFoundError:
    print("Archivo de audio no encontrado para el test.")
    sys.exit(1)

print("Enviando audio a la API (puede tardar por el Wavelet)...")
response = requests.post(url, files=files)

if response.status_code == 200:
    data = response.json()
    if data.get("success"):
        print("SUCCESSO!")
        print("Indicadores calculados:", data.get("indicators"))
        print("Gráficas generadas:", list(data["plots"].keys()))
        print("Imágenes generadas:", list(data["images"].keys()))
        print("Longitud de base64 espectrograma:", len(data["images"]["spectrogram"]))
        print("Longitud de base64 wavelet:", len(data["images"]["wavelet"]))
    else:
        print("La API devolvió un error:", data)
else:
    print("Error HTTP:", response.status_code)
    print("Respuesta:", response.text)
