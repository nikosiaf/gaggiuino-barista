import os
import time
import json
import threading
import subprocess
from datetime import datetime
from flask import Flask, jsonify
import requests

app = Flask(__name__)

# =========================
# CONFIG
# =========================
API_BASE        = os.getenv("API_BASE", "http://gaggiuino.local")
POLL_INTERVAL   = 3
SHOT_ID_POLL    = 5
POST_SHOT_DELAY = 8
MIN_SHOT_SECS   = 8
MAX_SHOT_SECS   = 180
TIMEOUT         = 5

HA_TOKEN          = os.getenv("SUPERVISOR_TOKEN", "")
HA_BASE           = "http://supervisor/core/api"
HA_NOTIFY_SERVICE = os.getenv("HA_NOTIFY_SERVICE", "notify.mobile_app_your_phone")

# =========================
# STATE
# =========================
state = {
    "known_shot_id":      None,
    "shot_running":       False,
    "shot_started_at":    None,
    "last_shot_ended_at": None,   # cooldown for pressure fallback trigger
    "last_plot":          None,
    "last_error":         None,
    "status":             "idle",
}

# =========================
# HELPERS
# =========================
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def send_notification(summary: dict, analysis: dict):
    """Send HA mobile notification from server.py where SUPERVISOR_TOKEN is valid."""
    if not HA_TOKEN:
        log("WARNING: SUPERVISOR_TOKEN not set - skipping notification")
        return

    # profile  = summary.get("profile", "unknown")
    duration = summary.get("duration_s", "")
    weight   = summary.get("final_weight_g", "")
    target_w = summary.get("target_weight_g", "")
    pressure = summary.get("max_pressure_bar", "")
    temp     = summary.get("water_temp_c", "")
    # provider = summary.get("ai_provider", "")
    title = "\u2615 Espresso Shot Done \u2615"
    
    # Build target weight string
    yield_str = f"{weight}g/{target_w}g" if target_w and target_w != "-" else f"{weight}g"

    lines = [
        # f"\u2615 {profile}",
        # f"\u23f1 {duration}s   \u2696 {yield_str}   \U0001f4c8 {pressure} bar   \U0001f321 {temp}\u00b0C",
        f"\U0001f321{temp}\u00b0C \U0001f4c8{pressure}bar \u2696{weight} \u23f1{duration}s",
    ]
    
    if analysis:
        # text  = analysis.get("notification_text", "")
        tuning = analysis.get("tuning", [])
        # verdict = analysis.get("verdict", "")
        shot_score = analysis.get("score")
        if shot_score:
            try:
                title = f"\u2615 Shot Score: {int(round(float(shot_score)))}/100 \u2615\n"
            except Exception:
                title = "Espresso Shot Done !"        
        # if verdict:
        #     lines.append(f"\n\U0001f9e0 {verdict}")
        if tuning:       
            # lines.append(f"\n\U0001f527 {tuning[:3]}")
            for tip in tuning[:1]:
                lines.append(f"\n\U0001f527 {tip}")
    else:
        lines.append("\n\U0001f916 AI analysis unavailable")

    url = f"{HA_BASE}/services/{HA_NOTIFY_SERVICE.replace('.', '/')}"
    payload = {
        # "title": "\u2615 Espresso Shot Done",
        "title": title,
        "message": "\n".join(lines),
        "data": {
            "image": "/local/gaggiuino-barista/last_shot.png",
            "push": {"sound": "default"},
        },
    }
    try:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {HA_TOKEN}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        log(f"Notification response: HTTP {r.status_code} - {r.text[:100]}")
    except Exception as e:
        log(f"WARNING: Notification failed: {e}")


def get_machine_status() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/api/system/status", timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            data = data[0]
        return {
            "pressure":    float(data.get("pressure", 0)),
            "brew_switch": bool(data.get("brewSwitchState", False)),
            "temperature": float(data.get("temperature", 0)),
            "weight":      float(data.get("weight", 0)),
            "profile":     data.get("profileName", "unknown"),
        }
    except Exception:
        return None


def get_latest_shot_id() -> int | None:
    try:
        r = requests.get(f"{API_BASE}/api/shots/latest", timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            data = data[0]
        shot_id = data.get("lastShotId") or data.get("id")
        return int(shot_id) if shot_id is not None else None
    except Exception:
        return None


def run_plot(shot_id: int, duration: float):
    log(f"Waiting {POST_SHOT_DELAY}s for Gaggiuino to finalize shot recording...")
    time.sleep(POST_SHOT_DELAY)
    log(f"Starting plot for shot #{shot_id} (duration ~{duration:.0f}s)...")
    state["status"] = "plotting"
    try:
        result = subprocess.run(
            ["python", "/app/src/plot_logic.py"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            state["last_plot"] = datetime.now().isoformat()
            state["last_error"] = None
            state["status"] = "idle"
            # Parse summary JSON from stdout for notification
            summary = {}
            analysis = {}
            for line in result.stdout.strip().splitlines():
                if line.startswith("SUMMARY:"):
                    try:
                        summary = json.loads(line[8:])
                        analysis = {
                            "verdict": summary.get("verdict", ""),
                            "tuning": summary.get("tuning", []),
                            "score": summary.get("score", 0),
                            "notification_text": summary.get("notification_text", ""),
                        }
                    except Exception:
                        log(f"  -> {line}")
                elif line.startswith("WARNING") or "Gemini" in line or "Anthropic" in line:
                    log(f"  [AI] {line}")
                else:
                    log(f"  -> {line}")
            log("Plot completed successfully.")
            send_notification(summary, analysis)
        else:
            state["last_error"] = result.stderr.strip()
            state["status"] = "error"
            log("Plot FAILED:")
            for line in result.stderr.strip().splitlines():
                log(f"  {line}")
    except subprocess.TimeoutExpired:
        state["last_error"] = "Plot timed out after 300s"
        state["status"] = "error"
        log("Plot TIMED OUT after 300s")
    except Exception as e:
        state["last_error"] = str(e)
        state["status"] = "error"
        log(f"Plot EXCEPTION: {e}")


# =========================
# WATCHER LOOP
# =========================
def watcher():
    log("Watcher started - detecting shots via brew switch + shot ID change...")
    offline = False

    while True:
        machine = get_machine_status()

        # --- offline handling ---
        if machine is None:
            if not offline:
                log("Machine unreachable - switching to 30s polling interval")
                offline = True
                state["status"] = "offline"
                state["shot_running"] = False
                state["shot_started_at"] = None
            time.sleep(30)
            continue

        if offline:
            log(f"Machine back online | profile={machine['profile']} temp={machine['temperature']:.1f}C")
            offline = False
            state["status"] = "idle"
            # Re-learn current shot ID so we don't false-trigger on startup
            state["known_shot_id"] = get_latest_shot_id()
            log(f"Current shot ID: {state['known_shot_id']}")

        pressure    = machine["pressure"]
        brew_switch = machine["brew_switch"]
        elapsed     = time.time() - state["shot_started_at"] if state["shot_started_at"] else 0

        # --- verbose log every 10s during active shot ---
        if state["shot_running"] and elapsed > 0 and int(elapsed) % 10 < POLL_INTERVAL:
            log(f"  [shot {elapsed:.0f}s] brew_switch={brew_switch} pressure={pressure:.2f}bar weight={machine['weight']:.1f}g")

        # --- shot START: brew switch turned on ---
        if not state["shot_running"] and brew_switch:
            state["shot_running"] = True
            state["shot_started_at"] = time.time()
            log(f"Shot STARTED (brew switch) | profile={machine['profile']} temp={machine['temperature']:.1f}C")

        # --- shot START fallback: pressure spike (brew switch unreliable) ---
        # Suppressed for 60s after a shot ends to avoid residual pressure triggers
        elif not state["shot_running"] and pressure >= 2.0:
            ended_at = state["last_shot_ended_at"]
            cooldown_elapsed = time.time() - ended_at if ended_at else 999
            if cooldown_elapsed < 60:
                pass  # silently ignore — residual pressure after shot
            else:
                state["shot_running"] = True
                state["shot_started_at"] = time.time()
                log(f"Shot STARTED (pressure {pressure:.2f}bar) | profile={machine['profile']}")

        # --- shot END: brew switch turned off ---
        elif state["shot_running"] and not brew_switch:
            duration = time.time() - state["shot_started_at"]
            state["shot_running"] = False
            state["shot_started_at"] = None
            state["last_shot_ended_at"] = time.time()
            log(f"Shot END signal | duration={duration:.0f}s pressure={pressure:.2f}bar")

            if duration < MIN_SHOT_SECS:
                log(f"Ignored - too short ({duration:.0f}s)")
            elif duration > MAX_SHOT_SECS:
                log(f"Ignored - too long ({duration:.0f}s)")
            else:
                # Wait for shot ID to increment - confirms Gaggiuino saved it
                log("Waiting for Gaggiuino to save shot record...")
                saved_id = _wait_for_new_shot_id(timeout=30)
                if saved_id:
                    log(f"New shot ID detected: #{saved_id} - triggering plot")
                    state["known_shot_id"] = saved_id
                    threading.Thread(
                        target=run_plot,
                        args=(saved_id, duration),
                        daemon=True
                    ).start()
                else:
                    log("WARNING: Shot ID did not increment within 30s - skipping plot")

        time.sleep(POLL_INTERVAL)


def _wait_for_new_shot_id(timeout: int = 30) -> int | None:
    """Poll until shot ID increments, return new ID or None on timeout."""
    deadline = time.time() + timeout
    current_id = state["known_shot_id"]
    while time.time() < deadline:
        new_id = get_latest_shot_id()
        if new_id is not None and (current_id is None or new_id > current_id):
            return new_id
        time.sleep(2)
    return None


# =========================
# FLASK ROUTES
# =========================
@app.route("/plot/latest", methods=["POST", "GET"])
def plot_latest():
    machine = get_machine_status()
    if machine is None:
        return jsonify({"ok": False, "error": "Gaggiuino unreachable - machine may be offline"}), 503
    shot_id = get_latest_shot_id()
    log(f"Manual plot triggered | shot_id={shot_id} profile={machine['profile']}")
    try:
        result = subprocess.run(
            ["python", "/app/src/plot_logic.py"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            log("Manual plot FAILED:")
            for line in result.stderr.strip().splitlines():
                log(f"  {line}")
            return jsonify({"ok": False, "stderr": result.stderr}), 500

        # Log all stdout lines so we can see AI analysis progress
        summary = {}
        analysis = {}
        for line in result.stdout.strip().splitlines():
            if line.startswith("SUMMARY:"):
                try:
                    summary = json.loads(line[8:])
                    analysis = {
                        "verdict":           summary.get("verdict", ""),
                        "tuning":            summary.get("tuning", []),
                        "score": summary.get("score", 0),
                        "notification_text": summary.get("notification_text", ""),
                    }
                    log(f"  [AI] Shot #{summary.get('shot_id')} | {summary.get('profile')} | verdict: {bool(summary.get('verdict'))}")
                except Exception:
                    log(f"  -> {line}")
            elif line.startswith("WARNING") or "Gemini" in line or "Anthropic" in line:
                log(f"  [AI] {line}")
            else:
                log(f"  -> {line}")

        log("Manual plot completed.")
        send_notification(summary, analysis)
        return jsonify({"ok": True, "stdout": result.stdout})
    except subprocess.TimeoutExpired:
        log("Manual plot TIMED OUT after 300s")
        return jsonify({"ok": False, "error": "Plot timed out"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/status", methods=["GET"])
def status():
    machine = get_machine_status()
    elapsed = time.time() - state["shot_started_at"] if state["shot_started_at"] else None
    return jsonify({
        "watcher_status": state["status"],
        "shot_running":   state["shot_running"],
        "shot_elapsed_s": round(elapsed, 1) if elapsed else None,
        "known_shot_id":  state["known_shot_id"],
        "last_plot":      state["last_plot"],
        "last_error":     state["last_error"],
        "machine":        machine,
    })


# =========================
# STARTUP
# =========================
import pathlib
import sys

_www_dir = pathlib.Path("/homeassistant/www/gaggiuino-barista")
if not _www_dir.exists():
    _www_dir.mkdir(parents=True, exist_ok=True)
    log(f"Created output directory: {_www_dir}")
else:
    log(f"Output directory exists: {_www_dir}")

# --- Check REST sensor is configured in HA ---
log("Checking HA REST sensor configuration...")
try:
    _sensor_check = requests.get(
        "http://supervisor/core/api/states/sensor.gaggiuino_barista_last_shot",
        headers={"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"},
        timeout=5,
    )
    if _sensor_check.status_code == 200:
        log("REST sensor OK: sensor.gaggiuino_barista_last_shot found")
    elif _sensor_check.status_code == 404:
        log("=" * 60)
        log("WARNING: REST sensor not configured!")
        log("sensor.gaggiuino_barista_last_shot does not exist in HA.")
        log("Please follow Step 1 in the add-on Documentation tab")
        log("to add the REST sensor to your configuration.yaml,")
        log("then restart Home Assistant and rebuild this add-on.")
        log("=" * 60)
        sys.exit(1)
    else:
        log(f"WARNING: Unexpected response checking REST sensor: HTTP {_sensor_check.status_code}")
except Exception as e:
    log(f"WARNING: Could not check REST sensor (Supervisor API unavailable?): {e}")

log("Checking machine status on startup...")
startup_status = get_machine_status()
if startup_status:
    log(f"Machine ONLINE | profile={startup_status['profile']} temp={startup_status['temperature']:.1f}C pressure={startup_status['pressure']:.2f}bar")
    startup_id = get_latest_shot_id()
    state["known_shot_id"] = startup_id
    log(f"Current shot ID: {startup_id}")
else:
    log("Machine OFFLINE at startup - watcher will detect when it comes back")

log("Starting watcher thread...")
threading.Thread(target=watcher, daemon=True).start()

log("Starting web server on port 5000...")
try:
    from waitress import serve
    log("Using waitress WSGI server")
    serve(app, host="0.0.0.0", port=5000)
except ImportError:
    log("waitress not found - falling back to Flask dev server")
    app.run(host="0.0.0.0", port=5000)