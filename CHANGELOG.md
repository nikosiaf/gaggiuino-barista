# Changelog

## 2.0.2

### Fixed
- **Connection Loss Handling**: Plot subprocess now handles Gaggiuino connection loss gracefully
  - Detects connection errors in plot output
  - Retries once after 5 seconds
  - If still unreachable, aborts gracefully without exposing exception details
  - Returns to polling instead of leaving status in "error" state

---

## 2.0.1

### Fixed
- **Shot Detection**: Now properly ignores steam and hot water modes
  - Added `steam_switch` from Gaggiuino API to detect steam/hot water mode
  - Added `water_level` check - ignores if water level < 10
  - Added profile name check - ignores if profile starts with "[UT]" (user test mode)
  - Shot detection now requires: brew_switch=ON AND steam_switch=OFF AND water_level>=10 AND profile not "[UT]"
- **Pressure Fallback Removed**: No longer uses pressure-based shot detection (unreliable)
- **API Fields Added**: `brewSwitchState` and `steamSwitchState` now properly read from `/api/system/status`

---

## 2.0.0

### Added
- **Language Selection**: New `llm_language` config option for AI phrasing (en, el, it, de, es, fr)
- **Profile-Aware Analysis**: Complete rewrite of annotation engine with profile-specific thresholds for lever, flow control, pressure, and filter profiles
- **Pre-infusion Analysis**: New `extract_preinfusion_phases()` detects wetting, soaking, compression phases with uniformity scoring
- **Flow Ratio Tracking**: New `extract_flow_ratio_metrics()` computes weight_flow/pump_flow ratio to detect channeling and restriction
- **Taste-Based Scoring**: New `taste_based_scoring()` produces taste profile (well_extracted/mostly_balanced/slightly_off/unbalanced/poorly_extracted)
- **Profile Matching**: Automatically matches shot profile against 41 bundled community profiles
  - Exact/fuzzy name matching
  - Phase structure matching when name doesn't match
- **Profile Adherence Scoring**: Scores how well shot followed matched profile expectations
- **Local Profiles Support**: Reads custom profiles from `/homeassistant/www/gaggiuino-barista/profiles/`
- **New Events**: slow_preinfusion, fast_preinfusion, uneven_preinfusion, possible_channeling, poor_profile_adherence

### Changed
- **Scoring Recalibrated**: Deterministic scoring now aligns with human evaluation
  - Target hit gives +10 bonus (was +5)
  - Stability thresholds raised - only extreme deviations penalized
  - Pre-infusion uniformity tolerance increased
  - Flow ratio thresholds relaxed
  - "well_extracted" threshold lowered (net >= 1, was >= 2)
  - Severity penalties reduced (2 per warning, was 3)
- **Score Thresholds Updated**: 80+ (very good), 70-79 (good with flaws), 60-69 (drinkable but off), <60 (problematic)
- **Score Stamp Changed**: "SHOT X/100" to "SCORE X/100"
- **LLM Prompt Enhanced**: Now includes profile context, extraction profile, and flow ratio for better phrasing
- **Fallback Analysis Improved**: Uses new extraction_profile for profile-specific tuning recommendations
- **New Output Fields**: profile_match_type, profile_match_confidence, matched_profile_name, profile_adherence_score, taste_profile, extraction_profile

### Fixed
- Empty taste_profile and extraction_profile fields in output
- Profile adherence scoring now checks real metrics (pressure/flow stability, flow ratio)
- Score and heuristic fields properly passed through to JSON output

### Technical
- 41 Gaggiuino community profiles bundled in Docker image at `/app/profiles/`
- Profiles directory added to Dockerfile COPY command
- Dynamic token limits: 700 tokens for English, 1500 tokens for other languages (UTF-8 multi-byte encoding requires more tokens)
- Environment variable `LLM_LANGUAGE` properly propagated to subprocess via `env=os.environ.copy()`
- Detailed code comments added to all source files (server.py, plot_logic.py, annotation_engine.py)

---

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

---

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
- Rolling shot history in `shot_history.json` (last 10 shots)
- Rolling PNG history (last 30 graphs) with automatic cleanup
- Manual trigger via HTTP `GET/POST /plot/latest` — triggers plot + notification
- Health check endpoint `GET /status` — returns watcher state + live machine data
- Automatic offline detection — switches to 30s polling when machine unreachable
- Output directory auto-created on startup if it doesn't exist
- Production WSGI server (waitress) — no Flask development server warning
- Supervisor API integration — `homeassistant_api: true`, no long-lived token required
- Port 5000 exposed via `ports: 5000/tcp: 5000` in config.yaml
