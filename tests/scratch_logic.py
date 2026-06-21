import urllib.request
import json
import time

payload = {
    "query_type": "type1",
    "premises-NL": [
        "If a drone has a high-quality camera, it has long battery life.",
        "Drone X does not have long battery life."
    ],
    "question": "Based on the above premises, which of the following is true?\nA. Drone X has a high-quality camera.\nB. Drone X does not have a high-quality camera."
}

req = urllib.request.Request(
    "http://localhost:8000/answer",
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

print("Running Contradiction test...")
with urllib.request.urlopen(req, timeout=300) as response:
    result_data = json.loads(response.read().decode('utf-8'))
    print(json.dumps(result_data, indent=2))
