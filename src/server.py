"""
Gaggiuino Barista - Server Module

This module runs as the main entry point for the Home Assistant add-on.
It provides:
- A background watcher that detects espresso shots from the Gaggiuino machine
- Shot detection via brew switch state + shot ID confirmation
- Flask web server for manual plot triggering and status queries
- Mobile push notifications via Home Assistant Companion app

The watcher polls the Gaggiuino machine every 3 seconds. When a shot is detected:
1. Waits for the shot to be saved (shot ID increments)
2. Spawns a subprocess to generate the shot graph and run AI analysis
3. Sends a mobile notification with the annotated graph

Shot detection logic:
- Primary: Brew switch turns ON -> OFF
- Fallback: Pressure >= 2.0 bar (for machines with unreliable brew switch)
- Suppression: Pressure fallback ignored for 60s after a shot ends (residual pressure)

Environment variables (set via config.yaml / run.sh):
- API_BASE: Gaggiuino API base URL (default: http://gaggiuino.local)
- SUPERVISOR_TOKEN: Home Assistant Supervisor API token
- HA_NOTIFY_SERVICE: HA notify service (e.g., notify.mobile_app_your_phone)
- LLM_LANGUAGE: Language for AI analysis (en, el, it, de, es, fr)
"""

import os
import pathlib
import time
import json
import threading
import hmac
import subprocess
import sys
from datetime import datetime
from flask import Flask, jsonify, request
from waitress import serve
import requests

from redact import redact


# =========================
# FLASK APP SETUP
# =========================
app = Flask(__name__)


# =========================
# CONFIGURATION
# =========================
# Polling intervals (seconds)
API_BASE        = os.getenv("API_BASE", "http://gaggiuino.local")
POLL_INTERVAL   = 3          # How often to check machine status during normal operation
SHOT_ID_POLL    = 5          # How often to poll for new shot ID while waiting for save
POST_SHOT_DELAY = 8          # Wait time after shot ends before fetching data (allows Gaggiuino to save)
MIN_SHOT_SECS   = 8          # Minimum shot duration to be considered valid (ignore flushes, etc.)
MAX_SHOT_SECS   = 180        # Maximum shot duration (ignore accidental left-on machines)
API_TOKEN       = os.getenv("API_TOKEN", "")  # Required by POST /plot/latest
WORKER_ENV      = ("PATH", "HOME", "TZ", "LANG", "LC_ALL", "API_BASE", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
                   "GEMINI_API_KEY", "GEMINI_MODEL", "LLM_LANGUAGE", "MPLCONFIGDIR", "MPLBACKEND")
TIMEOUT         = 5          # HTTP request timeout when talking to Gaggiuino

# Home Assistant integration
HA_TOKEN          = os.getenv("SUPERVISOR_TOKEN", "")
HA_BASE           = "http://supervisor/core/api"
HA_NOTIFY_SERVICE = os.getenv("HA_NOTIFY_SERVICE", "notify.mobile_app_your_phone")

# Language configuration for AI analysis
LLM_LANGUAGE  = os.getenv("LLM_LANGUAGE", "en")
LANGUAGES     = {"en": "English", "el": "Greek", "it": "Italian", "de": "German", "es": "Spanish", "fr": "French"}


# =========================
# WATCHER STATE
# =========================
_analysis_lock = threading.Lock()  # one analysis at a time: each can cost an AI call
# Shared state between watcher thread and web server
# Used to track shot detection progress and machine status
state = {
    "known_shot_id":      None,   # Last known shot ID (prevents re-triggering on old shots)
    "shot_running":        False,  # True when brew switch is on
    "shot_started_at":     None,   # Timestamp when current shot started
    "last_shot_ended_at":  None,   # Timestamp when last shot ended
    "last_plot":           None,   # ISO timestamp of last successful plot
    "last_error":          None,   # Last error message (for status endpoint)
    "status":              "idle", # Current status: idle, plotting, offline, error
}


# =========================
# LOGGING HELPER
# =========================
def log(msg: str):
    """Print timestamped log message to stdout (captured by Docker/HA logs)."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {redact(msg)}", flush=True)


# =========================
# HOME ASSISTANT NOTIFICATIONS
# =========================
def send_notification(summary: dict, analysis: dict):
    """
    Send mobile push notification via Home Assistant Companion app.
    
    Args:
        summary: Shot metadata dict (duration, weight, pressure, temp, profile)
        analysis: AI analysis dict (score, verdict, tuning, notification_text)
    
    The notification includes:
    - Title with shot score (e.g., "☕ Shot Score: 82/100 ☕")
    - Shot parameters (temp, pressure, weight, duration)
    - First tuning tip from AI analysis
    - Inline graph image
    
    Note: Called from server.py where SUPERVISOR_TOKEN is valid.
          plot_logic.py cannot send notifications directly (token not inherited).
    """
    # Check if we have a valid HA token
    if not HA_TOKEN:
        log("WARNING: SUPERVISOR_TOKEN not set - skipping notification")
        return

    # Extract shot parameters
    duration = summary.get("duration_s", "")
    weight   = summary.get("final_weight_g", "")
    target_w = summary.get("target_weight_g", "")
    pressure = summary.get("max_pressure_bar", "")
    temp     = summary.get("water_temp_c", "")
    
    # Build notification title (includes shot score if available)
    title = "\u2615 Espresso Shot Done \u2615"
    
    # Format yield string (e.g., "36g/38g" or just "36g" if no target)
    yield_str = f"{weight}g/{target_w}g" if target_w and target_w != "-" else f"{weight}g"

    # Build notification body
    lines = [
        f"\U0001f321{temp}\u00b0C \U0001f4c8{pressure}bar \u2696{yield_str} \u23f1{duration}s",
    ]
    
    # Add AI analysis results if available
    if analysis:
        tuning = analysis.get("tuning", [])
        shot_score = analysis.get("score")
        
        # Include score in title if available
        if shot_score:
            try:
                title = f"\u2615 Shot Score: {int(round(float(shot_score)))}/100 \u2615\n"
            except Exception:
                title = "Espresso Shot Done !"
        
        # Include first tuning tip
        if tuning:
            lines.append(f"\n\U0001f527 {tuning[0]}")
    else:
        lines.append("\n\U0001f916 AI analysis unavailable")

    # Build notification payload for HA API
    notify_url = f"{HA_BASE}/services/{HA_NOTIFY_SERVICE.replace('.', '/')}"
    
    # Resolve the graph URL - try Supervisor API first, fall back to relative path
    # The relative path works when accessed from within HA network
    # This shot's own file: last_shot.png would show a newer shot once there is one
    graph_path = f"/local/gaggiuino-barista/{summary.get('graph_file', 'last_shot.png')}"
    ha_base_url = os.getenv("HA_BASE_URL", "").rstrip("/")
    
    # Try to get HA base URL from Supervisor config
    if not ha_base_url:
        try:
            cfg = requests.get(
                "http://supervisor/core/api/config",
                headers={"Authorization": f"Bearer {HA_TOKEN}"},
                timeout=5,
            ).json()
            # Use external_url (public) or internal_url (local) as base
            ha_base_url = (cfg.get("external_url") or cfg.get("internal_url") or "").rstrip("/")
        except Exception as e:
            log(f"WARNING: Could not fetch HA base URL from Supervisor: {e}")

    # Build full graph URL if we have a base, otherwise use relative path
    graph_url = f"{ha_base_url}{graph_path}" if ha_base_url else graph_path

    # Construct the notification payload
    payload = {
        "title": title,
        "message": "\n".join(lines),
        "data": {
            "image": graph_path,      # Relative path for HA Companion app
            "url": graph_url,         # Full URL for tapping notification
            "push": {"sound": "default"},
        },
    }
    
    # Send the notification via HA API
    try:
        response = requests.post(
            notify_url,
            headers={
                "Authorization": f"Bearer {HA_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=10,
        )
        log(f"Notification response: HTTP {response.status_code} - {response.text[:100]}")
    except Exception as e:
        log(f"WARNING: Notification failed: {e}")


# =========================
# GAGGIUINO API HELPERS
# =========================
def get_machine_status() -> dict | None:
    """
    Fetch current machine status from Gaggiuino API.
    
    Returns:
        Dict with pressure, brew_switch, steam_switch, water_level, temperature, weight, profile
        or None if machine is unreachable.
    """
    try:
        response = requests.get(f"{API_BASE}/api/system/status", timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        # Handle both list and dict responses
        if isinstance(data, list):
            data = data[0]
        
        return {
            "pressure":     float(data.get("pressure", 0)),
            "brew_switch":  bool(data.get("brewSwitchState", False)),
            "steam_switch": bool(data.get("steamSwitchState", False)),
            "water_level":  int(data.get("waterLevel", 0)),
            "temperature":  float(data.get("temperature", 0)),
            "weight":       float(data.get("weight", 0)),
            "profile":      data.get("profileName", "unknown"),
        }
    except Exception:
        return None


def get_latest_shot_id() -> int | None:
    """
    Get the most recent shot ID from Gaggiuino.
    
    Returns:
        Integer shot ID, or None if unavailable.
    """
    try:
        response = requests.get(f"{API_BASE}/api/shots/latest", timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        # Handle both list and dict responses
        if isinstance(data, list) and data:
            data = data[0]
        
        # Extract shot ID from various possible field names
        shot_id = data.get("lastShotId") or data.get("id")
        return int(shot_id) if shot_id is not None else None
    except Exception:
        return None


# =========================
# SHOT PLOTTING
# =========================
def _handle_plot_failure(result: subprocess.CompletedProcess, shot_id: int):
    """
    Handle plot subprocess failure with connection-aware retry logic.
    
    - Connection errors: retry once after 5s, then abort gracefully
    - Other errors: log stderr lines for debugging
    """
    stderr = result.stderr.strip()
    is_connection_error = any(
        kw in stderr for kw in ("Connection", "Timeout", "timed out", "ConnectTimeout")
    )
    
    if is_connection_error:
        log("Connection lost during plot - retrying in 5s...")
        time.sleep(5)
        
        retry = _run_analysis(shot_id)
        
        if retry.returncode == 0:
            state["last_plot"] = datetime.now().isoformat()
            state["last_error"] = None
            state["status"] = "idle"
            _parse_and_notify(retry.stdout)
            log("Plot completed successfully on retry.")
            return
        else:
            log("Plot aborted - machine unreachable")
            state["last_error"] = "Connection unavailable"
            state["status"] = "idle"
            return
    else:
        state["last_error"] = redact(stderr.splitlines()[-1] if stderr else "Plot failed")[:300]
        state["status"] = "error"
        log("Plot FAILED:")
        for line in stderr.splitlines():
            log(f"  {line}")


def _parse_and_notify(stdout: str):
    """Parse SUMMARY: JSON from stdout and send notification."""
    summary = {}
    analysis = {}
    for line in stdout.strip().splitlines():
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
                pass
        elif line.startswith("WARNING") or "Gemini" in line or "Anthropic" in line:
            log(f"  [AI] {line}")
        else:
            log(f"  -> {line}")
    
    send_notification(summary, analysis)
    return summary


def _worker_env() -> dict:
    """What the analysis process gets: only what it uses. Not the Supervisor token."""
    return {k: os.environ[k] for k in WORKER_ENV if k in os.environ}


def _run_analysis(shot_id: int) -> subprocess.CompletedProcess:
    """Graph and analyze exactly this shot."""
    return subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_logic.py"),
         "--shot-id", str(shot_id)],
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout for plot generation
        env=_worker_env(),
    )


def run_plot(shot_id: int, duration: float):
    """
    Spawn subprocess to generate shot graph and run AI analysis.
    
    This runs in a separate thread to avoid blocking the watcher.
    The subprocess runs plot_logic.py which:
    1. Fetches complete shot data from Gaggiuino
    2. Generates the shot graph image
    3. Runs deterministic telemetry analysis
    4. Calls AI (Anthropic/Gemini) for phrasing
    5. Saves graph with AI annotations
    6. Writes JSON data files
    7. Prints SUMMARY: JSON to stdout
    
    Args:
        shot_id: The Gaggiuino shot ID to plot
        duration: Shot duration in seconds (for logging)
    """
    log(f"Waiting {POST_SHOT_DELAY}s for Gaggiuino to finalize shot recording...")
    time.sleep(POST_SHOT_DELAY)
    log(f"Starting plot for shot #{shot_id} (duration ~{duration:.0f}s)...")
    
    state["status"] = "plotting"
    
    try:
        with _analysis_lock:
            result = _run_analysis(shot_id)
        
        if result.returncode == 0:
            state["last_plot"] = datetime.now().isoformat()
            state["last_error"] = None
            state["status"] = "idle"
            _parse_and_notify(result.stdout)
            log("Plot completed successfully.")
        else:
            _handle_plot_failure(result, shot_id)

    except subprocess.TimeoutExpired:
        state["last_error"] = "Plot timed out after 300s"
        state["status"] = "error"
        log("Plot TIMED OUT after 300s")
        
    except Exception as e:
        state["last_error"] = redact(e)[:300]
        state["status"] = "error"
        log(f"Plot EXCEPTION: {e}")


# =========================
# SHOT DETECTION WATCHER
# =========================
def watcher():
    """
    Background thread that continuously polls Gaggiuino for shot detection.
    
    Detection logic:
    1. Brew switch ON + steam switch OFF + water level >= 10 -> shot starts
    2. Brew switch OFF -> shot ends
    
    Ignored states:
    - water_level < 10 (low water)
    - steam_switch == True (steam or hot water mode)
    
    Waits for shot ID to increment before triggering plot (confirms save).
    
    Runs forever in a daemon thread until container stops.
    """
    log("Watcher started - detecting shots via brew switch + shot ID change...")
    offline = False

    while True:
        machine = get_machine_status()

        # --- Handle offline machine ---
        if machine is None:
            if not offline:
                log("Machine unreachable - switching to 30s polling interval")
                offline = True
                state["status"] = "offline"
                state["shot_running"] = False
                state["shot_started_at"] = None
            time.sleep(30)
            continue

        # --- Machine came back online ---
        if offline:
            log(f"Machine back online | profile={machine['profile']} temp={machine['temperature']:.1f}C")
            offline = False
            state["status"] = "idle"
            # Re-learn current shot ID to prevent false trigger on startup
            state["known_shot_id"] = get_latest_shot_id()
            log(f"Current shot ID: {state['known_shot_id']}")

        pressure     = machine["pressure"]
        brew_switch = machine["brew_switch"]
        steam_switch = machine["steam_switch"]
        water_level = machine["water_level"]
        elapsed     = time.time() - state["shot_started_at"] if state["shot_started_at"] else 0

        # --- Verbose logging during active shot (every 10s) ---
        if state["shot_running"] and elapsed > 0 and int(elapsed) % 10 < POLL_INTERVAL:
            log(f"  [shot {elapsed:.0f}s] brew_switch={brew_switch} pressure={pressure:.2f}bar weight={machine['weight']:.1f}g")

        # --- IGNORE: User test mode (profile starts with [UT]) ---
        if not state["shot_running"] and machine['profile'].startswith("[UT]"):
            pass  # Silent ignore - user test mode

        # --- IGNORE: Low water level ---
        elif not state["shot_running"] and water_level < 10:
            pass  # Silent ignore - low water

        # --- IGNORE: Steam or hot water mode ---
        elif not state["shot_running"] and steam_switch:
            pass  # Silent ignore - steam or hot water

        # --- SHOT START: Brew switch ON + steam OFF + water OK ---
        elif not state["shot_running"] and brew_switch and not steam_switch:
            state["shot_running"] = True
            state["shot_started_at"] = time.time()
            log(f"Shot STARTED | profile={machine['profile']} temp={machine['temperature']:.1f}C")

        # --- SHOT END: Brew switch OFF ---
        elif state["shot_running"] and not brew_switch:
            duration = time.time() - state["shot_started_at"]
            state["shot_running"] = False
            state["shot_started_at"] = None
            state["last_shot_ended_at"] = time.time()
            log(f"Shot END | duration={duration:.0f}s pressure={pressure:.2f}bar")

            # Validate shot duration
            if duration < MIN_SHOT_SECS:
                log(f"Ignored - too short ({duration:.0f}s)")
            elif duration > MAX_SHOT_SECS:
                log(f"Ignored - too long ({duration:.0f}s)")
            else:
                # Wait for shot ID to increment - confirms Gaggiuino saved the shot
                log("Waiting for Gaggiuino to save shot record...")
                saved_id = _wait_for_new_shot_id(timeout=30)
                if saved_id:
                    log(f"New shot ID detected: #{saved_id} - triggering plot")
                    state["known_shot_id"] = saved_id
                    # Run plot in separate thread to avoid blocking watcher
                    threading.Thread(
                        target=run_plot,
                        args=(saved_id, duration),
                        daemon=True
                    ).start()
                else:
                    log("WARNING: Shot ID did not increment within 30s - skipping plot")

        time.sleep(POLL_INTERVAL)


def _wait_for_new_shot_id(timeout: int = 30) -> int | None:
    """
    Poll until the shot ID increments (confirming Gaggiuino saved the shot).
    
    This prevents triggering plot before Gaggiuino has finished writing the shot data.
    
    Args:
        timeout: Maximum seconds to wait for new shot ID (default 30s)
    
    Returns:
        New shot ID if detected, or None on timeout.
    """
    deadline = time.time() + timeout
    current_id = state["known_shot_id"]
    
    while time.time() < deadline:
        new_id = get_latest_shot_id()
        if new_id is not None and (current_id is None or new_id > current_id):
            return new_id
        time.sleep(2)  # Poll every 2s while waiting
    
    return None


# =========================
# FLASK WEB ROUTES
# =========================
@app.route("/plot/latest", methods=["POST"])
def plot_latest():
    """
    Analyze the latest shot on request (a dashboard button or automation).

    POST /plot/latest with the header "Authorization: Bearer <api_token>".
    Every run can cost an AI call and sends a notification, so it needs the token from the
    add-on's configuration; without a configured token the endpoint stays off.

    Returns:
        JSON with ok, shot_id, score and profile; errors carry a short message, never process output.
    """
    if not API_TOKEN:
        return jsonify({"ok": False, "error": "Set api_token in the add-on configuration to use this endpoint"}), 403
    sent = request.headers.get("Authorization", "")
    if not hmac.compare_digest(sent.encode(), f"Bearer {API_TOKEN}".encode()):
        return jsonify({"ok": False, "error": "Missing or wrong token"}), 401

    machine = get_machine_status()
    if machine is None:
        return jsonify({"ok": False, "error": "Gaggiuino unreachable - machine may be offline"}), 503
    shot_id = get_latest_shot_id()
    if shot_id is None:
        return jsonify({"ok": False, "error": "The machine reports no shot"}), 503
    if not _analysis_lock.acquire(blocking=False):
        return jsonify({"ok": False, "error": "An analysis is already running"}), 409

    log(f"Manual plot triggered | shot_id={shot_id} profile={machine['profile']}")
    try:
        result = _run_analysis(shot_id)
        if result.returncode != 0:
            log("Manual plot FAILED:")
            for line in result.stderr.strip().splitlines():
                log(f"  {line}")
            return jsonify({"ok": False, "error": "Analysis failed, see the add-on log"}), 500
        summary = _parse_and_notify(result.stdout)
        log("Manual plot completed.")
        return jsonify({"ok": True, "shot_id": summary.get("shot_id"), "score": summary.get("score"),
                        "profile": summary.get("profile")})
    except subprocess.TimeoutExpired:
        log("Manual plot TIMED OUT after 300s")
        return jsonify({"ok": False, "error": "Analysis timed out"}), 500
    except Exception as e:
        log(f"Manual plot EXCEPTION: {e}")
        return jsonify({"ok": False, "error": "Analysis failed, see the add-on log"}), 500
    finally:
        _analysis_lock.release()


@app.route("/status", methods=["GET"])
def status():
    """
    Health check and status endpoint.
    
    GET /status
    
    Returns current watcher state, machine status, and last plot info.
    Useful for:
    - Dashboard sensors
    - Troubleshooting
    - Automation triggers
    
    Returns:
        JSON with watcher_status, shot_running, machine data, etc.
    """
    machine = get_machine_status()
    elapsed = time.time() - state["shot_started_at"] if state["shot_started_at"] else None
    
    return jsonify({
        "watcher_status": state["status"],
        "shot_running":   state["shot_running"],
        "shot_elapsed_s": round(elapsed, 1) if elapsed else None,
        "known_shot_id":  state["known_shot_id"],
        "last_plot":      state["last_plot"],
        "last_error":     redact(state["last_error"]) if state["last_error"] else None,
        "machine":        machine,
    })


# =========================
# STARTUP SEQUENCE
# =========================
def start():
    """Check the surroundings and start the watcher. Nothing here stops the add-on."""

    # Ensure output directory exists
    _www_dir = pathlib.Path("/homeassistant/www/gaggiuino-barista")
    if not _www_dir.exists():
        _www_dir.mkdir(parents=True, exist_ok=True)
        log(f"Created output directory: {_www_dir}")
    else:
        log(f"Output directory exists: {_www_dir}")

    # Verify REST sensor is configured in Home Assistant
    # This sensor reads last_shot.json and exposes it as an entity for dashboards
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
            log("then restart Home Assistant. Shot detection keeps running without it;")
            log("only the dashboard cards need it.")
            log("=" * 60)
        else:
            log(f"WARNING: Unexpected response checking REST sensor: HTTP {_sensor_check.status_code}")
    except Exception as e:
        log(f"WARNING: Could not check REST sensor (Supervisor API unavailable?): {e}")

    # Log configured language
    log(f"LLM Language: {LANGUAGES.get(LLM_LANGUAGE, LLM_LANGUAGE)}")

    # Check machine connectivity
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


if __name__ == "__main__":
    start()
    log("Starting web server on port 5000...")
    serve(app, host="0.0.0.0", port=5000)
