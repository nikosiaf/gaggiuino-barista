#!/usr/bin/env python3
"""
Command-line sensor script for Home Assistant.
Reads shot_history.json and finds each shot's own graph.
Outputs a single JSON object for use as a HA command_line sensor.
Place this file at: /config/scripts/gaggiuino_barista_history.py
Run with: python3 /config/scripts/gaggiuino_barista_history.py
"""
import json
from pathlib import Path

BASE_DIR  = Path("/config/www/gaggiuino-barista")
WEB_PATH  = "/local/gaggiuino-barista"
HISTORY_JSON = BASE_DIR / "shot_history.json"
MAX_SHOTS = 10


def graph_of(shot):
    """The graph of this very shot, by its ID. Files from before 2.1.0 are named shot_<time>_id<N>.png."""
    shot_id = shot.get("shot_id")
    for path in [BASE_DIR / f"shot_{shot_id}.png", *BASE_DIR.glob(f"shot_*_id{shot_id}.png")]:
        if path.exists():
            return {"shot_id": shot_id, "filename": path.name, "url": f"{WEB_PATH}/{path.name}",
                    "modified": int(path.stat().st_mtime)}
    return {"shot_id": shot_id, "filename": "", "url": "", "modified": 0}


def main():
    result = {
        "shots": [],
        "graphs": [],  # same order as shots: graphs[i] belongs to shots[i]
        "total_shots": 0,
    }

    if HISTORY_JSON.exists():
        try:
            shots = json.loads(HISTORY_JSON.read_text())
            result["shots"]       = shots[:MAX_SHOTS]
            result["total_shots"] = len(shots)
            result["graphs"]      = [graph_of(s) for s in result["shots"]]
        except Exception as e:
            result["error_json"] = str(e)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
