import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import requests
import logging
import sys

logger = logging.getLogger(__name__)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from annotation_engine import extract_features, detect_events, classify_extraction_tendency, summarize_for_prompt


# =========================
# THEME
# =========================
DARK_BG = "#061225"
HEADER_BG = "#0c1830"
GRID_COLOR = "#1a2942"
SPINE_COLOR = "#2b3b56"
TEXT_LIGHT = "#cbd5e1"
TEXT_DIM = "#94a3b8"
TEXT_BOX = "#e5e7eb"

plt.rcParams["figure.facecolor"] = DARK_BG
plt.rcParams["axes.facecolor"] = DARK_BG
plt.rcParams["savefig.facecolor"] = DARK_BG
plt.rcParams["savefig.edgecolor"] = DARK_BG
plt.rcParams["font.size"] = 10


# =========================
# CONFIG
# =========================
API_BASE = os.getenv("API_BASE", "http://gaggiuino.local")
OUT_DIR = Path("/homeassistant/www/gaggiuino-barista")
LAST_FILE = OUT_DIR / "last_shot.png"
TIMEOUT = 10
MAX_HISTORY = 30

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HA_TOKEN = os.getenv("SUPERVISOR_TOKEN", "")
HA_BASE = "http://supervisor/core/api"
HA_NOTIFY_SERVICE = os.getenv("HA_NOTIFY_SERVICE", "notify.mobile_app_your_phone")


# =========================
# COLORS
# =========================
COLOR_TEMP = "#ff5a2f"
COLOR_TEMP_TARGET = "#ff5a2f"

COLOR_PRESSURE = "#3b82f6"
COLOR_PRESSURE_TARGET = "#60a5fa"

COLOR_FLOW = "#facc15"
COLOR_FLOW_TARGET = "#eab308"

COLOR_WEIGHT_FLOW = "#22c55e"
COLOR_WEIGHT = "#a855f7"


# =========================
# HELPERS
# =========================
def get_json(url: str):
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def scale_list(values, factor=1.0):
    return [v / factor for v in values]


def cleanup_old_history_files():
    history_files = sorted(
        OUT_DIR.glob("shot_*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    for old_file in history_files[MAX_HISTORY:]:
        try:
            old_file.unlink()
        except Exception as e:
            print(f"WARNING: Could not delete {old_file}: {e}")


def cumulative_phase_times(phases):
    """
    Build phase boundary list from profile phases.
    For phases with a time stopCondition: use that time.
    For phases without time, skip them for background shading.
    """
    boundaries = []
    elapsed = 0.0
    for phase in phases:
        stop = phase.get("stopConditions", {})
        stop_time = stop.get("time")
        name = phase.get("name", "")
        if stop_time:
            elapsed += stop_time / 1000.0
            boundaries.append((elapsed, name))
    return boundaries


def moving_average(data, window=3):
    if not data or window <= 1:
        return data[:]
    out = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        chunk = data[start:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def clean_pump_flow(data):
    """Clip bad startup spikes without affecting the rest of the shot."""
    if not data:
        return data

    cleaned = data[:]
    early_points = min(12, len(cleaned))
    for i in range(early_points):
        if cleaned[i] > 6.0:
            cleaned[i] = 6.0

    return cleaned


def glow_plot(ax, x, y, color, lw=2.0, alpha=1.0, linestyle="-", zorder=3):
    ax.plot(x, y, color=color, linewidth=lw + 5.5, alpha=0.05, linestyle=linestyle, zorder=zorder - 2)
    ax.plot(x, y, color=color, linewidth=lw + 3.0, alpha=0.09, linestyle=linestyle, zorder=zorder - 1)
    ax.plot(x, y, color=color, linewidth=lw, alpha=alpha, linestyle=linestyle, zorder=zorder)


# =========================
# AI ANALYSIS
# =========================
_GEMINI_MIN_INTERVAL = 70
_GEMINI_LOCK_FILE = OUT_DIR / ".gemini_last_call"

SEVERITY_COLORS = {
    "good": "#22c55e",
    "info": "#60a5fa",
    "warning": "#facc15",
    "critical": "#ef4444",
}


def _gemini_get_last_call() -> float:
    try:
        return float(_GEMINI_LOCK_FILE.read_text().strip())
    except Exception:
        return 0.0


def _gemini_set_last_call():
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        _GEMINI_LOCK_FILE.write_text(str(time.time()))
    except Exception as e:
        print(f"WARNING: Could not write Gemini lock file: {e}")


def _strip_fenced_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _normalize_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _parse_ai_response(raw: str, duration_s: float, events: list, fallback: dict) -> dict:
    """Parse and validate JSON response from any AI provider."""
    result = json.loads(_strip_fenced_json(raw))
    valid_severities = set(SEVERITY_COLORS.keys())
    anchored_events = {round(float(e["time"]), 1): e for e in events}

    clean_annotations = []
    for ann in result.get("annotations", [])[:5]:
        ann_time = round(max(0.0, min(duration_s, _normalize_float(ann.get("time"), 0.0))), 1)
        if anchored_events:
            ann_time = min(anchored_events.keys(), key=lambda t: abs(t - ann_time))
            event = anchored_events[ann_time]
        else:
            event = {"reason": "Detected extraction event.", "type": "event", "severity": "info"}

        severity = ann.get("severity", event.get("severity", "info"))
        if severity not in valid_severities:
            severity = event.get("severity", "info")

        label = str(ann.get("label", event.get("type", "Event"))).strip()[:48] or "Event"
        reason = str(ann.get("reason", event.get("reason", "Detected extraction event."))).strip()[:180]

        clean_annotations.append({
            "time": ann_time,
            "label": label,
            "severity": severity,
            "reason": reason,
            "event_type": event.get("type", "event"),
        })

    parsed = {
        "score": max(0, min(100, int(result.get("score", fallback.get("score", 0))))),
        "confidence": max(
            0.0,
            min(
                1.0,
                _normalize_float(
                    result.get("confidence", fallback.get("confidence", 0.0)),
                    fallback.get("confidence", 0.0),
                ),
            ),
        ),
        "verdict": str(result.get("verdict", fallback.get("verdict", ""))).strip(),
        "tuning": [str(item).strip() for item in result.get("tuning", fallback.get("tuning", [])) if str(item).strip()][:3],
        "notification_text": str(result.get("notification_text", fallback.get("notification_text", ""))).strip(),
        "annotations": clean_annotations if clean_annotations else fallback.get("annotations", []),
    }

    if not parsed["verdict"]:
        parsed["verdict"] = fallback.get("verdict", "")
    if not parsed["tuning"]:
        parsed["tuning"] = fallback.get("tuning", [])
    if not parsed["notification_text"]:
        parsed["notification_text"] = fallback.get("notification_text", parsed["verdict"])

    return parsed


def _build_series_for_analysis(shot_data: dict) -> dict:
    dp = shot_data.get("datapoints", {})
    t = scale_list(dp.get("timeInShot", []), 10.0)
    return {
        "time_s": t,
        "pressure_bar": moving_average(scale_list(dp.get("pressure", []), 10.0), 2),
        "pump_flow_ml_s": moving_average(clean_pump_flow(scale_list(dp.get("pumpFlow", []), 10.0)), 3),
        "shot_weight_g": moving_average(scale_list(dp.get("shotWeight", []), 10.0), 2),
        "temp_c": moving_average(scale_list(dp.get("temperature", []), 10.0), 2),
        "weight_flow_g_s": moving_average(scale_list(dp.get("weightFlow", [0] * len(t)), 10.0), 3),
    }


def _build_fallback_analysis(features: dict, events: list, heuristic: dict) -> dict:
    event_labels = {
        "late_first_drops": "Late first drops",
        "early_first_drops": "Fast opening",
        "first_drops_on_time": "Good first drops",
        "high_peak_pressure": "High pressure peak",
        "low_peak_pressure": "Gentle pressure peak",
        "stable_core": "Stable core",
        "unstable_pressure": "Pressure wobble",
        "unstable_flow": "Flow wobble",
        "restricted_flow": "Restricted core",
        "fast_core_flow": "Fast core flow",
        "tail_runaway": "Tail opens up",
        "tail_controlled": "Clean tail",
        "target_hit": "Target hit",
        "stopped_early": "Stopped early",
        "ran_past_target": "Ran long",
    }

    chosen_events = events[:4]
    annotations = []
    for event in chosen_events:
        annotations.append({
            "time": event["time"],
            "label": event_labels.get(event["type"], event["type"].replace("_", " ").title()),
            "severity": event["severity"],
            "reason": event["reason"],
            "event_type": event["type"],
        })

    tendency = heuristic["tendency"]
    score = heuristic["score_hint"]
    verdict = (
        f"Score {score}/100: {features['profile_name']} looks {tendency}, "
        f"with the main story coming from first drops, core stability, and the tail."
    )

    tuning = []
    if any(e["type"] == "late_first_drops" for e in events):
        tuning.append("Grind slightly coarser or reduce puck resistance a touch to bring first drops earlier.")
    elif any(e["type"] == "early_first_drops" for e in events):
        tuning.append("Grind slightly finer to slow the opening and build more body.")

    if any(e["type"] in {"unstable_pressure", "unstable_flow", "tail_runaway"} for e in events):
        tuning.append("Improve puck prep and stop the shot 1–2g earlier if the tail opens again.")
    else:
        tuning.append("Keep the same overall profile shape and only make small grind adjustments shot to shot.")

    if any(e["type"] == "stopped_early" for e in events):
        tuning.append("Let the shot run slightly longer to reach the target yield.")
    elif any(e["type"] == "ran_past_target" for e in events):
        tuning.append("Stop the shot a little earlier to protect sweetness in the tail.")

    tuning = tuning[:3]
    notification_text = verdict
    if tuning:
        notification_text += " Next shot: " + " ".join(tuning[:2])

    return {
        "score": score,
        "confidence": heuristic["confidence_hint"],
        "verdict": verdict,
        "tuning": tuning,
        "notification_text": notification_text[:280],
        "annotations": annotations,
    }


def _build_llm_prompt(features: dict, events: list, heuristic: dict, fallback: dict) -> str:
    compact = summarize_for_prompt(features, events, heuristic)
    return f"""You are an expert espresso barista assisting a Home Assistant add-on.

This system already ran a deterministic telemetry analyzer. Your job is NOT to rediscover events from raw data. Your job is to:
- phrase the detected events clearly
- write a concise overall verdict
- suggest 2-3 actionable next-shot tweaks
- return a final score from 0 to 100
- preserve the provided event times as anchors

Respond with valid JSON only. No markdown. No extra text.

Input JSON:
{json.dumps(compact, indent=2)}

Return exactly this JSON schema:
{{
  "score": <integer 0-100>,
  "confidence": <float 0.0-1.0>,
  "verdict": "<one sentence overall assessment>",
  "tuning": [
    "<actionable recommendation 1>",
    "<actionable recommendation 2>"
  ],
  "annotations": [
    {{
      "time": <float; must match one of the provided detected event times>,
      "label": "<short label max 4 words>",
      "severity": "<good|info|warning|critical>",
      "reason": "<short explanation up to 18 words>"
    }}
  ],
  "notification_text": "<2 short sentences for mobile push>"
}}

Rules:
- Use only the detected events already provided. Do not invent new times.
- Keep 3 to 5 annotations total.
- Keep labels short and readable on a graph.
- Tuning must be practical for the next shot.
- Align the score with the analyzer hints unless there is a strong reason to move it slightly.
- Score hint: {heuristic['score_hint']}
- Confidence hint: {heuristic['confidence_hint']}
- Fallback verdict style reference: {fallback['verdict']}
"""


def _analyze_with_anthropic(prompt: str, duration_s: float, events: list, fallback: dict) -> dict:
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            "max_tokens": 700,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    if not response.ok:
        logger.warning("Anthropic status=%s body=%s", response.status_code, response.text)
    response.raise_for_status()
    raw = response.json()["content"][0]["text"].strip()
    result = _parse_ai_response(raw, duration_s, events, fallback)
    result["provider"] = "anthropic"
    return result


def _analyze_with_gemini(prompt: str, duration_s: float, events: list, fallback: dict) -> dict:
    since_last = time.time() - _gemini_get_last_call()
    if since_last < _GEMINI_MIN_INTERVAL:
        wait = _GEMINI_MIN_INTERVAL - since_last
        print(f"Gemini rate limiter: waiting {wait:.0f}s before calling API...")
        time.sleep(wait)

    _gemini_set_last_call()
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 700},
        },
        timeout=30,
    )
    if response.status_code == 429:
        error_body = response.json() if response.content else {}
        error_msg = str(error_body).lower()
        if "quota" in error_msg or "day" in error_msg:
            print("WARNING: Gemini daily quota exceeded - AI unavailable until quota resets (midnight PT)")
        else:
            print("WARNING: Gemini per-minute rate limit hit - backing off")
        _gemini_set_last_call()
        return {}
    response.raise_for_status()
    raw = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    result = _parse_ai_response(raw, duration_s, events, fallback)
    result["provider"] = "gemini"
    return result


def analyze_shot_with_ai(shot_data: dict) -> dict:
    """Deterministic analysis always; LLM phrasing when provider exists."""
    series = _build_series_for_analysis(shot_data)
    features = extract_features(shot_data, series)
    events = detect_events(features)
    heuristic = classify_extraction_tendency(features, events)
    fallback = _build_fallback_analysis(features, events, heuristic)

    fallback["provider"] = "deterministic"
    fallback["features"] = features
    fallback["detected_events"] = events
    fallback["heuristic"] = heuristic

    if not ANTHROPIC_API_KEY and not GEMINI_API_KEY:
        print("AI provider not configured - using deterministic analysis")
        return fallback

    prompt = _build_llm_prompt(features, events, heuristic, fallback)
    result = {}

    if ANTHROPIC_API_KEY:
        try:
            print("AI phrasing via Anthropic Claude...")
            result = _analyze_with_anthropic(prompt, features["duration_s"], events, fallback)
            print("Anthropic analysis OK")
        except Exception as e:
            print(f"WARNING: Anthropic analysis failed: {e} - falling back to Gemini")
            result = {}

    if not result and GEMINI_API_KEY:
        try:
            print("AI phrasing via Gemini...")
            result = _analyze_with_gemini(prompt, features["duration_s"], events, fallback)
            if result:
                print("Gemini analysis OK")
        except Exception as e:
            print(f"WARNING: Gemini analysis failed: {e}")
            result = {}

    if not result:
        return fallback

    result["features"] = features
    result["detected_events"] = events
    result["heuristic"] = heuristic
    return result


def _draw_score_stamp(ax, analysis: dict):
    """Draw a stable faux rubber-stamp score in the top-left corner."""
    if not analysis:
        return

    score = analysis.get("score")
    if score is None:
        return

    try:
        score_int = int(round(float(score)))
    except Exception:
        return

    stamp_text = f"SHOT {score_int}/100"

    if score_int >= 85:
        color = "#39FF14"
    elif score_int >= 70:
        color = "#CFFF04"
    elif score_int >= 55:
        color = "#FFB000"
    else:
        color = "#FF3131"

    x = 0.05
    y = 0.76
    rotation = 22

    ax.text(
        x,
        y,
        stamp_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        rotation=rotation,
        rotation_mode="anchor",
        fontsize=24,
        fontweight="bold",
        color=color,
        alpha=0.22,
        zorder=58,
        clip_on=False,
        bbox=dict(
            boxstyle="square,pad=0.30",
            facecolor="none",
            edgecolor=color,
            linewidth=4.8,
        ),
    )

    ax.text(
        x,
        y,
        stamp_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        rotation=rotation,
        rotation_mode="anchor",
        fontsize=24,
        fontweight="bold",
        color=color,
        alpha=0.96,
        zorder=59,
        clip_on=False,
        bbox=dict(
            boxstyle="square,pad=0.20",
            facecolor="none",
            edgecolor=color,
            linewidth=2.4,
        ),
    )

    ax.text(
        x - 0.0015,
        y + 0.0015,
        stamp_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        rotation=rotation,
        rotation_mode="anchor",
        fontsize=24,
        fontweight="bold",
        color=color,
        alpha=0.06,
        zorder=57,
        clip_on=False,
    )


# =========================
# GRAPH ANNOTATIONS OVERLAY
# =========================
def draw_annotations(ax_press, ax_temp, t, pressure, pump_flow, annotations: list):
    """Draw timestamped annotation arrows on the chart."""
    if not annotations or not t:
        return

    press_max = ax_press.get_ylim()[1]
    used_y_positions = []

    for ann in annotations:
        ann_t = float(ann.get("time", 0))
        label = ann.get("label", "")
        severity = ann.get("severity", "info")
        color = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["info"])

        if t:
            idx = min(range(len(t)), key=lambda i: abs(t[i] - ann_t))
            y_data = pressure[idx] if idx < len(pressure) else press_max * 0.5
        else:
            y_data = press_max * 0.5

        base_y = min(y_data + 1.5, press_max * 0.88)
        for used_y in used_y_positions:
            if abs(base_y - used_y) < 0.9:
                base_y = used_y + 1.0
        base_y = min(base_y, press_max * 0.92)
        used_y_positions.append(base_y)

        ax_press.annotate(
            label,
            xy=(ann_t, y_data),
            xytext=(ann_t, base_y),
            fontsize=12,  #fontsize=7.5,
            color=color,
            ha="center",
            va="bottom",
            zorder=20,
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.1,
                alpha=0.85,
            ),
            bbox=dict(
                facecolor=DARK_BG,
                edgecolor=color,
                alpha=0.82,
                pad=2.0,
                boxstyle="round,pad=0.3",
            ),
        )


def _score_color(score) -> str:
    try:
        score_int = int(round(float(score)))
    except Exception:
        return TEXT_LIGHT

    if score_int >= 85:
        return "#39FF14"
    if score_int >= 70:
        return "#CFFF04"
    if score_int >= 55:
        return "#FFB000"
    return "#FF3131"


def draw_verdict_panel(fig, analysis: dict):
    """Top-left verdict panel."""
    if not analysis:
        return

    verdict = (analysis.get("verdict") or "").strip()
    score = analysis.get("score")

    if not verdict:
        return

    import textwrap
    color = _score_color(score)
    panel_text = textwrap.fill(verdict, width=140)

    fig.text(
        0.066,
        0.875,
        panel_text,
        ha="left",
        va="bottom",
        fontsize=12,
        color=color,
        zorder=10,
        family="monospace",
        bbox=dict(
            facecolor="#0a1628",
            edgecolor="#2b3b56",
            alpha=0.90,
            pad=7.5,
            boxstyle="round,pad=0.5",
        ),
    )


def draw_tuning_panel(fig, analysis: dict):
    """Bottom-right tuning panel."""
    if not analysis:
        return

    tuning = analysis.get("tuning", []) or []
    score = analysis.get("score")

    if not tuning:
        return

    import textwrap
    color = _score_color(score)

    wrapped_lines = []
    for tip in tuning[:1]:
    # for tip in tuning[0]:    
        wrapped_lines.append("★ " + textwrap.fill(tip, width=140)) #52
    panel_text = "\n".join(wrapped_lines)

    fig.text(
        0.066, #0.62,
        0.064,
        panel_text,
        ha="left",
        va="bottom",
        fontsize=12,
        color=color,
        zorder=10,
        family="monospace",
        bbox=dict(
            facecolor="#0a1628",
            edgecolor="#2b3b56",
            alpha=0.90,
            pad=7.5,
            boxstyle="round,pad=0.5",
        ),
    )


def write_shot_json(summary: dict, analysis: dict):
    """Write full shot data + analysis to last_shot.json and shot_history.json."""
    try:
        shot_json = {
            "datetime": datetime.now().strftime("%d/%m/%Y-%H:%M:%S"),
            "shot_id": summary.get("shot_id"),
            "profile": summary.get("profile", ""),
            "duration_s": summary.get("duration_s"),
            "final_weight_g": summary.get("final_weight_g"),
            "target_weight_g": summary.get("target_weight_g"),
            "max_pressure_bar": summary.get("max_pressure_bar"),
            "water_temp_c": summary.get("water_temp_c"),
            "history_count": summary.get("history_count"),
            "ai_available": bool(analysis),
            "ai_provider": analysis.get("provider", "") if analysis else "",
            "score": analysis.get("score") if analysis else None,
            "confidence": analysis.get("confidence") if analysis else None,
            "verdict": analysis.get("verdict", "") if analysis else "",
            "tuning": analysis.get("tuning", []) if analysis else [],
            "notification_text": analysis.get("notification_text", "") if analysis else "",
            "annotations": analysis.get("annotations", []) if analysis else [],
            "features": analysis.get("features", {}) if analysis else {},
            "detected_events": analysis.get("detected_events", []) if analysis else [],
        }

        json_file = OUT_DIR / "last_shot.json"
        json_file.write_text(json.dumps(shot_json, indent=2))
        print(f"Shot JSON written to {json_file}")

        history_file = OUT_DIR / "shot_history.json"
        try:
            history = json.loads(history_file.read_text()) if history_file.exists() else []
        except Exception:
            history = []

        history = [s for s in history if s.get("shot_id") != shot_json["shot_id"]]
        history.insert(0, shot_json)
        history = history[:5]
        history_file.write_text(json.dumps(history, indent=2))
        print(f"Shot history updated ({len(history)} shots)")

    except Exception as e:
        print(f"WARNING: Could not write shot JSON: {e}")

# =========================
# MAIN
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    latest = get_json(f"{API_BASE}/api/shots/latest")

    if isinstance(latest, list) and latest:
        latest_item = latest[0]
    elif isinstance(latest, dict):
        latest_item = latest
    else:
        raise RuntimeError(f"Unexpected response from /api/shots/latest: {latest}")

    shot_id = latest_item.get("lastShotId") or latest_item.get("id")
    if shot_id is None:
        raise RuntimeError(f"Could not find shot id in response: {latest}")

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    history_file = OUT_DIR / f"shot_{timestamp_str}_id{shot_id}.png"

    shot = get_json(f"{API_BASE}/api/shots/{shot_id}")
    dp = shot["datapoints"]

    # ---------- raw data ----------
    t = scale_list(dp["timeInShot"], 10.0)

    pressure = scale_list(dp["pressure"], 10.0)
    target_pressure = scale_list(dp["targetPressure"], 10.0)

    pump_flow = scale_list(dp["pumpFlow"], 10.0)
    target_pump_flow = scale_list(dp["targetPumpFlow"], 10.0)

    temp = scale_list(dp["temperature"], 10.0)
    target_temp = scale_list(dp["targetTemperature"], 10.0)

    shot_weight = scale_list(dp["shotWeight"], 10.0)
    raw_weight_flow = scale_list(dp.get("weightFlow", [0] * len(t)), 10.0)

    # ---------- cleanup / smoothing ----------
    pressure = moving_average(pressure, 2)
    target_pressure = moving_average(target_pressure, 2)

    pump_flow = clean_pump_flow(pump_flow)
    pump_flow = moving_average(pump_flow, 3)
    target_pump_flow = moving_average(target_pump_flow, 2)

    weight_flow = moving_average(raw_weight_flow, 3)

    temp = moving_average(temp, 2)
    target_temp = moving_average(target_temp, 2)

    shot_weight = moving_average(shot_weight, 2)

    # ---------- metadata ----------
    profile = shot.get("profile", {})
    phases = profile.get("phases", [])
    phase_boundaries = cumulative_phase_times(phases)

    duration_raw = shot.get("duration", 0)
    duration_s = duration_raw / 10.0 if duration_raw else (t[-1] if t else 0)
    final_weight = shot_weight[-1] if shot_weight else 0
    max_pressure = max(pressure) if pressure else 0
    profile_name = profile.get("name", "Unknown Profile")
    target_final_weight = profile.get("globalStopConditions", {}).get("weight", "-")
    water_temp = profile.get("waterTemperature", "-")

    # =========================
    # FIGURE / AXES
    # =========================
    fig = plt.figure(figsize=(16, 9), facecolor=DARK_BG)

    ax_temp = fig.add_subplot(111)
    ax_press = ax_temp.twinx()
    ax_grams = ax_temp.twinx()

    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax_temp, ax_press, ax_grams):
        ax.set_facecolor(DARK_BG)

    for spine in ax_grams.spines.values():
        spine.set_visible(False)
    ax_grams.set_yticks([])
    ax_grams.set_yticklabels([])
    ax_grams.set_ylabel("")
    ax_grams.tick_params(left=False, right=False, labelleft=False, labelright=False)

    # =========================
    # HEADER STRIP
    # =========================
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    header_text = (
        f"{now_str}   |   {profile_name}   |   "
        f"TIME {duration_s:.1f}s   |   "
        f"YIELD {final_weight:.1f}g / {target_final_weight}g   |   "
        f"TEMP {water_temp}\u00b0C   |   "
        f"PEAK {max_pressure:.1f} bar"
    )

    header_bar = plt.Rectangle(
        (0.01, 0.94),
        0.98,
        0.06,
        transform=fig.transFigure,
        facecolor=HEADER_BG,
        edgecolor="#1f2d45",
        linewidth=1.0,
        zorder=2
    )
    fig.patches.append(header_bar)

    fig.text(
        0.5, 0.97,
        header_text,
        ha="center",
        va="center",
        fontsize=11,
        color=TEXT_LIGHT,
        zorder=3
    )

    # =========================
    # PHASE BACKGROUNDS
    # =========================
    phase_colors = [
        "#0e1a2f",
        "#0b1629",
        "#101d33",
        "#0c1a30",
        "#0a172a",
        "#0f1c31",
    ]

    start_x = 0.0
    for idx, (end_x, _name) in enumerate(phase_boundaries):
        ax_temp.axvspan(
            start_x,
            end_x,
            color=phase_colors[idx % len(phase_colors)],
            alpha=0.18,
            linewidth=0,
        )
        start_x = end_x

    PREINFUSION_KEYWORDS = {
        "wetting", "wet", "quick wet", "bloom", "pre-infusion", "preinfusion",
        "pre infusion", "compression", "compress", "puck compression",
        "compression ramp", "ramp", "short cr", "quick wetting",
    }

    total_t = max(t) if t else 30
    split1 = 0.0
    split2 = total_t * 0.80

    if phase_boundaries:
        named = [(xv, n) for xv, n in phase_boundaries if n.strip()]
        if named:
            for end_x, name in named:
                if any(kw in name.lower() for kw in PREINFUSION_KEYWORDS):
                    split1 = end_x
            if split1 == 0.0:
                split1 = phase_boundaries[0][0]
        else:
            split1 = phase_boundaries[0][0]

    if split2 <= split1 + 2:
        split2 = split1 + (total_t - split1) * 0.65

    def draw_divider(xv):
        ax_temp.axvline(xv, linestyle="--", linewidth=1.2, alpha=0.45, color="#7c8ea8", zorder=5)

    def draw_stage_label(x_start, x_end, label):
        ax_temp.text(
            (x_start + x_end) / 2.0,
            86.5,
            label,
            color=TEXT_DIM,
            fontsize=8,
            ha="center",
            va="center",
            alpha=0.95,
        )

    if split1 > 0:
        draw_divider(split1)
        draw_divider(split2)
        draw_stage_label(0, split1, "Pre-Infusion")
        draw_stage_label(split1, split2, "Main Extraction")
        draw_stage_label(split2, total_t, "Final Phase")
    else:
        split1 = total_t * 0.20
        split2 = total_t * 0.80
        draw_divider(split1)
        draw_divider(split2)
        draw_stage_label(0, split1, "Pre-Infusion")
        draw_stage_label(split1, split2, "Main Extraction")
        draw_stage_label(split2, total_t, "Final Phase")

    # =========================
    # PLOTS
    # =========================
    glow_plot(ax_temp, t, temp, COLOR_TEMP, lw=1.8, zorder=5)
    glow_plot(ax_temp, t, target_temp, COLOR_TEMP_TARGET, lw=1.4, linestyle=(0, (4, 3)), alpha=0.7, zorder=4)

    glow_plot(ax_press, t, pressure, COLOR_PRESSURE, lw=2.8, zorder=8)
    glow_plot(ax_press, t, target_pressure, COLOR_PRESSURE_TARGET, lw=1.8, linestyle=(0, (4, 3)), alpha=0.7, zorder=7)

    glow_plot(ax_press, t, pump_flow, COLOR_FLOW, lw=2.2, zorder=6)
    glow_plot(ax_press, t, target_pump_flow, COLOR_FLOW_TARGET, lw=1.5, linestyle=(0, (4, 3)), alpha=0.7, zorder=5)

    glow_plot(ax_press, t, weight_flow, COLOR_WEIGHT_FLOW, lw=2.0, zorder=5)
    glow_plot(ax_grams, t, shot_weight, COLOR_WEIGHT, lw=2.6, zorder=6)

    # =========================
    # LIMITS / AXES
    # =========================
    ax_temp.set_xlim(left=0, right=max(t) if t else 30)
    ax_temp.set_ylim(0, 100)
    ax_temp.set_yticks(range(0, 101, 20))

    ax_press.set_ylim(0, 10)
    ax_press.set_yticks(range(0, 11, 2))

    ax_grams.set_ylim(0, max(60, max(shot_weight, default=0) + 8))

    ax_temp.set_xlabel("Time (s)", fontsize=11, color=TEXT_LIGHT)
    ax_temp.set_ylabel("Temperature (\u00b0C)", fontsize=11, color=COLOR_TEMP)
    ax_press.set_ylabel("Pressure / Flow (bar / ml/s)", fontsize=11, color=COLOR_PRESSURE)

    ax_temp.tick_params(axis="x", colors=TEXT_LIGHT)
    ax_temp.tick_params(axis="y", colors=COLOR_TEMP)
    ax_press.tick_params(axis="y", colors=COLOR_PRESSURE)

    ax_temp.grid(True, which="major", axis="both", color=GRID_COLOR, linestyle="-", alpha=0.6)

    for ax in (ax_temp, ax_press):
        ax.spines["top"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color(SPINE_COLOR)

    # =========================
    # YIELD LABEL
    # =========================
    if t:
        ax_temp.axvline(t[-1], linestyle=":", linewidth=1.1, alpha=0.35, color="#90a4c0")
        ax_grams.text(
            t[-1] - 0.98,
            final_weight,
            f"{final_weight:.1f}gr",
            ha="right",
            va="center",
            fontsize=10,
            color=TEXT_BOX,
            bbox=dict(
                facecolor="#111827",
                edgecolor=SPINE_COLOR,
                alpha=0.9,
                pad=0.2,
            ),
        )

    # =========================
    # LEGEND
    # =========================
    legend_elements = [
        Line2D([0], [0], color=COLOR_TEMP_TARGET, lw=1.6, linestyle=(0, (4, 3)), label="Target Temperature"),
        Line2D([0], [0], color=COLOR_TEMP, lw=1.8, label="Temperature (\u00b0C)"),
        Line2D([0], [0], color=COLOR_PRESSURE_TARGET, lw=2.0, linestyle=(0, (4, 3)), label="Target Pressure"),
        Line2D([0], [0], color=COLOR_PRESSURE, lw=2.8, label="Pressure"),
        Line2D([0], [0], color=COLOR_FLOW_TARGET, lw=1.8, linestyle=(0, (4, 3)), label="Target Flow"),
        Line2D([0], [0], color=COLOR_FLOW, lw=2.4, label="Pump Flow"),
        Line2D([0], [0], color=COLOR_WEIGHT_FLOW, lw=2.0, label="Weight Flow"),
        Line2D([0], [0], color=COLOR_WEIGHT, lw=2.6, label="Shot Weight"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=8,
        frameon=False,
        fontsize=10,
        labelcolor=TEXT_LIGHT,
        handlelength=3,
    )

    plt.tight_layout(rect=[0.015, 0.08, 0.985, 0.92])

    # =========================
    # SAVE GRAPH (without AI first)
    # =========================
    def save_figure():
        fig.savefig(
            LAST_FILE,
            dpi=175,
            bbox_inches="tight",
            facecolor=DARK_BG,
            edgecolor=DARK_BG,
        )

    save_figure()
    print("Graph saved (no AI overlay yet)")

    # =========================
    # AI ANALYSIS + OVERLAY
    # =========================
    analysis = analyze_shot_with_ai(shot)

    if analysis:
        draw_verdict_panel(fig, analysis)
        draw_tuning_panel(fig, analysis)

        annotations = analysis.get("annotations", [])
        if annotations:
            draw_annotations(ax_press, ax_temp, t, pressure, pump_flow, annotations)

        _draw_score_stamp(ax_temp, analysis)
        save_figure()
        print("Graph re-saved with AI annotations")
    else:
        print("AI analysis unavailable - graph saved without annotations")

    plt.close(fig)

    if not LAST_FILE.exists():
        raise RuntimeError(f"Chart was not created: {LAST_FILE}")

    shutil.copy2(LAST_FILE, history_file)

    if not history_file.exists():
        raise RuntimeError(f"History chart was not created: {history_file}")

    cleanup_old_history_files()

    summary = {
        "ok": True,
        "shot_id": shot_id,
        "last_file": str(LAST_FILE),
        "history_file": str(history_file),
        "history_count": len(list(OUT_DIR.glob("shot_*.png"))),
        "duration_s": round(duration_s, 1),
        "final_weight_g": round(final_weight, 1),
        "target_weight_g": target_final_weight,
        "max_pressure_bar": round(max_pressure, 1),
        "water_temp_c": water_temp,
        "profile": profile_name,
    }

    if analysis:
        summary["ai_provider"] = analysis.get("provider", "")
        summary["score"] = analysis.get("score")
        summary["confidence"] = analysis.get("confidence")
        summary["verdict"] = analysis.get("verdict", "")
        summary["tuning"] = analysis.get("tuning", [])
        summary["notification_text"] = analysis.get("notification_text", "")

    write_shot_json(summary, analysis)
    print("SUMMARY:" + json.dumps(summary))


if __name__ == "__main__":
    main()