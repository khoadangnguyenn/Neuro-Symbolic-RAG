import urllib.request, json
req = urllib.request.Request(
    "http://localhost:8000/answer",
    data=json.dumps({"query_type": "type2", "question": "At the three vertices of right-angled triangle ABC (right-angled at A), with AB = 30 cm, AC = 40 cm, and BC = 50 cm, charges q1 = q2 = q3 = 2x10^-9 C are placed. Determine the magnitude of the net electric force acting on a charge q = -2x10^-9 C placed at point H, which is the foot of the altitude from A"}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
try:
    with urllib.request.urlopen(req, timeout=300) as response:
        res = json.loads(response.read().decode('utf-8'))
        if "metadata" in res and "execution_errors" in res["metadata"]:
            print("EXECUTION ERRORS:")
            for err in res["metadata"]["execution_errors"]:
                print("  ->", err)
        else:
            print("SUCCESS! No execution errors found.")
            print("Answer:", res.get("answer"))
except Exception as e:
    print("Error connecting to backend:", e)
