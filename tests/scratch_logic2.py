import urllib.request
import json
import time

payload = {
    "query_type": "type1",
    "premises-NL": [
        "If it rains, the ground gets wet.",
        "It is raining today."
    ],
    "question": "Based on the premises, which statement is true?\nA. The ground is dry\nB. The ground is wet\nC. It is snowing"
}

req = urllib.request.Request(
    "http://localhost:8000/answer",
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

print("Running Choose True test...")
with urllib.request.urlopen(req, timeout=300) as response:
    result_data = json.loads(response.read().decode('utf-8'))
    print(json.dumps(result_data, indent=2))
