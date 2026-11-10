from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def main():
    request_body = (ROOT / "sample_request.json").read_bytes()
    request = Request(
        f"{API_URL}/predict",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))

    expected = json.loads((ROOT / "expected_predictions.json").read_text(encoding="utf-8"))
    actual = result["predictions"]
    if not np.allclose(actual, expected, rtol=1e-8, atol=1e-8):
        raise AssertionError(f"API predictions differ from exported model: {actual} != {expected}")

    print("PASS: API predictions match the exported model.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
