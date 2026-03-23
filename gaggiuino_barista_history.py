#!/usr/bin/env python3
"""
Command-line sensor script for Home Assistant.
Reads shot_history.json and last 5 PNG files from the gaggiuino-barista output dir.
Outputs a single JSON object for use as a HA command_line sensor.
Place this file at: /config/scripts/gaggiuino_barista_history.py
Run with: python3 /config/scripts/gaggiuino_barista_history.py
"""
import json
import os
from pathlib import Path

BASE_DIR  = Path("/config/www/gaggiuino-barista")
WEB_PATH  = "/local/gaggiuino-barista"
HISTORY_JSON = BASE_DIR / "shot_history.json"

def main():
    result = {
        "shots": [],
        "graphs": [],
        "total_shots": 0,
    }

    # Load shot_history.json (last 5 shots data)
    if HISTORY_JSON.exists():
        try:
            shots = json.loads(HISTORY_JSON.read_text())
            result["shots"]       = shots[:6]
            result["total_shots"] = len(shots)
        except Exception as e:
            result["error_json"] = str(e)

    # List last 5 PNG graph files sorted by modification time
    if BASE_DIR.exists():
        try:
            pngs = sorted(
                BASE_DIR.glob("shot_*.png"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )[:5]
            result["graphs"] = [
                {
                    "filename": p.name,
                    "url":      f"{WEB_PATH}/{p.name}",
                    "modified": int(p.stat().st_mtime),
                }
                for p in pngs
            ]
        except Exception as e:
            result["error_graphs"] = str(e)

    print(json.dumps(result))

if __name__ == "__main__":
    main()