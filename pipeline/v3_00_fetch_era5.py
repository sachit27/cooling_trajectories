"""Fetch the model-pinned ERA5 cloud inputs used by v3_10.

This recipe intentionally sets ``models=era5``. Open-Meteo's response JSON does
not echo the model name, so the exact request parameters are retained here and in
``data_external/era5_pinned/README.md``. Existing files are replaced atomically
only after the response schema and requested time span have been validated.
"""
from __future__ import annotations

import json
import os
import tempfile
import urllib.parse
import urllib.request


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data_external", "era5_pinned")
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
POINTS = {
    "centre": (47.3769, 8.5417),
    "nw": (47.43, 8.47),
    "ne": (47.43, 8.61),
    "sw": (47.32, 8.47),
    "se": (47.32, 8.61),
}
COMMON = {
    "start_date": "2020-06-01",
    "end_date": "2025-08-31",
    "hourly": ("cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
               "temperature_2m,wind_speed_10m"),
    "timezone": "UTC",
    "models": "era5",
    "cell_selection": "nearest",
}
EXPECTED_FIRST = "2020-06-01T00:00"
EXPECTED_LAST = "2025-08-31T23:00"


def validate(payload: dict) -> None:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times or times[0] != EXPECTED_FIRST or times[-1] != EXPECTED_LAST:
        raise ValueError("ERA5 response has an unexpected hourly time span")
    for variable in ["cloud_cover", "cloud_cover_low", "wind_speed_10m"]:
        values = hourly.get(variable)
        if values is None or len(values) != len(times):
            raise ValueError(f"missing or truncated hourly variable: {variable}")


def main() -> None:
    os.makedirs(DEST, exist_ok=True)
    for name, (lat, lon) in POINTS.items():
        query = {"latitude": lat, "longitude": lon, **COMMON}
        url = BASE_URL + "?" + urllib.parse.urlencode(query)
        print(f"fetching {name}: {url}")
        request = urllib.request.Request(
            url, headers={"User-Agent": "zurich-night-heat-reproducibility/1.0"})
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read()
        payload = json.loads(raw)
        validate(payload)
        target = os.path.join(DEST, f"openmeteo_era5_zurich_{name}.json")
        fd, temporary = tempfile.mkstemp(prefix="era5_", suffix=".json", dir=DEST)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(f"  wrote {target} ({len(raw):,} bytes)")


if __name__ == "__main__":
    main()
