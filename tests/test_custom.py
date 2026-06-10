import urllib.request
import json
payload = {
    "query_type": "type2",
    "question": "Calculate the energy stored in capacitor C when C = 100 μF and U = 50 V."
}
req = urllib.request.Request(
    "http://localhost:8000/answer",
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
with urllib.request.urlopen(req) as response:
    print(response.read().decode('utf-8'))
