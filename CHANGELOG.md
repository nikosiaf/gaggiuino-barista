# Changelog

## 1.1.0

### Added
- New hybrid annotation engine for shot analysis
- Deterministic telemetry feature extraction before any LLM call
- Deterministic event detection with severity and time anchors
- LLM phrasing layer that rewrites detected events into clean labels, verdict, tuning, and `0-100` score
- Shared schema contract between Anthropic primary and Gemini fallback
- New AI metadata in `last_shot.json`: `ai_provider`, `score`, `confidence`, `features`, `detected_events`

### Changed
- README and documentation now describe Anthropic primary + Gemini fallback correctly
- AI annotations are now grounded to detected event times instead of free-form timestamps
- Add-on description updated for deterministic + AI architecture

## 1.0.4

### Added
- Anthropic Claude API support as preferred AI provider
- Set `anthropic_api_key` in add-on configuration to use Claude Haiku
- Reliable, no rate limits, approximately $0.001 per shot analysis
- Falls back to Gemini free tier if Anthropic key is not set
- Falls back gracefully to no AI if neither key is configured
- Shared prompt and response parsing logic between providers
- Clear log messages indicating which AI provider is being used

---

## 1.0.3

### Fixed
- Add preinfusion and Extraction phases in graph.

---

## 1.0.2

### Fixed
- False shot starts from residual pressure after shot ends
- Pressure fallback trigger now suppressed for 60s after brew switch turns off
- Eliminates spurious "Shot STARTED (pressure X.XXbar)" log entries during machine depressurization
- Ignored short shots from residual pressure no longer appear in logs

---

## 1.0.1

### Added
- Startup check for `sensor.gaggiuino_barista_last_shot` REST sensor via Supervisor API
- If sensor is not configured in HA, add-on logs a clear warning and exits
- Forces user to complete Step 1 of documentation before the add-on will run

---

## 1.0.0

### Added
- Automatic shot detection via brew switch + shot ID confirmation
- Watcher polls Gaggiuino every 3s detecting brew switch state
- After brew switch off, waits for shot ID to increment before triggering plot
- This guarantees plot runs only after Gaggiuino has fully saved the shot record
- Fallback: pressure >= 2.0 bar triggers shot start if brew switch is unreliable
- Shots shorter than 8s or longer than 180s are ignored (flushes, machine left on)
- Espresso shot graph generation with dark theme and glow-effect curves
- Multi-axis chart: temperature, pressure, pump flow, weight flow, shot weight
- Phase background shading and labels (preinfusion, compression, extraction, decline)
- Header strip with shot metadata (datetime, profile, duration, yield, temp, peak pressure)
- Yield label annotation at end of shot
- Google Gemini AI analysis (free tier)
- Structured JSON response: verdict, timestamped annotations, tuning recommendations
- AI annotation overlays on graph at key moments (color-coded: good/info/warning/critical)
- AI summary panel inside graph image
- Graph always saved immediately — AI overlays added on top if analysis succeeds
- File-based rate limiter: max 1 Gemini call per 70s, persists across subprocess restarts
- Distinguishes per-minute vs daily quota 429 errors
- Mobile push notification via HA Companion app
- Sent from server.py using SUPERVISOR_TOKEN (no long-lived token needed)
- Includes profile, duration, yield, peak pressure, temperature, AI analysis
- Includes graph image inline
- JSON shot data written to `/homeassistant/www/gaggiuino-barista/last_shot.json`
- Rolling shot history in `shot_history.json` (last 5 shots)
- Rolling PNG history (last 30 graphs) with automatic cleanup
- Manual trigger via HTTP `GET/POST /plot/latest` — triggers plot + notification
- Health check endpoint `GET /status` — returns watcher state + live machine data
- Automatic offline detection — switches to 30s polling when machine unreachable
- Output directory auto-created on startup if it doesn't exist
- Production WSGI server (waitress) — no Flask development server warning
- Supervisor API integration — `homeassistant_api: true`, no long-lived token required
- Port 5000 exposed via `ports: 5000/tcp: 5000` in config.yaml
