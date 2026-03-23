![Logo](https://github.com/nikosiaf/gaggiuino-barista/blob/main/logo.png)

# Gaggiuino Barista

Automatically generates espresso shot graphs with a hybrid deterministic + AI annotation engine and sends mobile push notifications via Home Assistant after every shot.

**Version:** 1.1.0

---

## How it works

The add-on runs a background watcher polling your Gaggiuino machine every 3 seconds. After a shot is detected and confirmed saved by Gaggiuino, it:

1. Fetches the complete shot data
2. Generates a detailed graph and saves it immediately
3. Runs a deterministic telemetry analyzer to compute extraction features and detect events
4. Calls Anthropic to phrase those events into labels, verdict, tuning and score (Gemini fallback uses the same schema)
5. Re-saves the graph with AI annotations if analysis succeeds
6. Writes shot data to `last_shot.json` and `shot_history.json`
7. Sends a mobile push notification with the graph and AI recommendations

---

## AI architecture in 1.1.0

The annotation engine now has two layers:

1. **Deterministic telemetry analyzer**
   - computes extraction features from pressure / flow / weight / temperature
   - detects events like late first drops, stable core, tail runaway, target hit/miss
   - assigns severity and time anchors before any LLM is called

2. **LLM phrasing layer**
   - Anthropic primary, Gemini fallback
   - rewrites detected events into short labels for the graph
   - produces a verdict, tuning recommendations, confidence, and `0-100` score
   - uses the same JSON schema for both providers

This makes the output more stable because the model no longer has to infer the shot structure from raw arrays alone.

## ⚠️ REQUIRED: Manual steps in Home Assistant

These steps **must** be completed before the add-on works correctly.

---

### Step 1 — Add REST sensor for last shot data

Add to Home Assistant `configuration.yaml` the following lines.
Replace `http://192.168.X.X:8123` with your actual Home Assistant IP address, or name and port (usually available uder: Settings --> System --> Network --> Home Assistant URL):

```yaml
rest:
  - resource: "http://192.168.X.X:8123/local/gaggiuino-barista/last_shot.json"
    scan_interval: 10
    sensor:
      - name: "Gaggiuino Barista Last Shot"
        unique_id: gaggiuino_barista_last_shot
        value_template: "{{ value_json.shot_id }}"
        json_attributes:
          - datetime
          - shot_id
          - profile
          - duration_s
          - final_weight_g
          - target_weight_g
          - max_pressure_bar
          - water_temp_c
          - history_count
          - ai_available
          - ai_provider
          - score
          - confidence
          - verdict
          - tuning
          - notification_text
          - annotations
          - features
          - detected_events
```

Restart HA. 
A new Entity will be created: `sensor.gaggiuino_barista_last_shot`

---

### Step 2 — Find your phone notify service name

Go to **Settings → Developer Tools → Actions**, search for `notify` and find the entry matching your phone. It looks like `notify.mobile_app_johns_iphone`. Use this exact value in the add-on **Configuration** tab.

---

### Step 3 — Get an AI API key

**Option A — Anthropic (recommended primary):** Go to https://console.anthropic.com, create an account, and generate an API key. Anthropic is now used as the phrasing/reasoning layer on top of the deterministic telemetry analyzer.

**Option B — Google Gemini (fallback):** Go to https://aistudio.google.com, sign in with a Google account, and click **Get API key → Create API key**. Gemini uses the exact same prompt contract and response schema as Anthropic, but free-tier limits can still cause AI analysis to be skipped.

You can set both keys — Anthropic is used first, Gemini only if Anthropic is unavailable or fails.

---

### Step 4 — Configure the add-on

In the add-on **Configuration** tab, set:

| Field | Value |
|-------|-------|
| `api_base` | http://gaggiuino.local or Gaggiuino IP address, incase mDNS issues, e.g. `http://192.168.1.100` |
| `anthropic_api_key` | Anthropic API key from Step 3A |
| `gemini_api_key` | Your Gemini API key from Step 3B |
| `ha_notify_service` | Your notify service from Step 2, e.g. `notify.mobile_app_johns_iphone` |

---

### Step 5 — Rebuild the add-on

After saving configuration, click **Rebuild** (not just Restart). 
HA re-reads permissions on rebuild — required for notifications to work.

---

## 📱 Mobile notification format

```
☕ Shot Score: 82/100
☕ Freddo 18-54
🌡93°C 📈9.4bar ⚖36g ⏱34s

🔧 Stop 1–2g earlier if the tail opens again

```
The graph image is attached inline.
---

## ➕ OPTIONAL: Dashboard sensors and cards

These steps add live shot data and graph history to your HA dashboard. Not required for the add-on core functionality.

---

### Optional Step A — Add rest_command for manual trigger from HA

If you want to trigger a plot manually from an HA dashboard button or automation, add to `configuration.yaml`:

```yaml
rest_command:
  gaggiuino_barista_plot_latest:
    url: "http://127.0.0.1:5000/plot/latest"
    method: POST
```

You can then call it from **Developer Tools → Actions** → `rest_command.gaggiuino_barista_plot_latest`, or add a button card to your dashboard:

```yaml
type: button
name: Plot Latest Shot
icon: mdi:coffee
tap_action:
  action: call-service
  service: rest_command.gaggiuino_barista_plot_latest
```

## ✅ Verify it works

After rebuilding:

1. Check the **Log** tab — you should see:
   ```
   Machine ONLINE | profile=... temp=...C pressure=...bar
   ```
2. Open `http://your-ha-ip:5000/status` — you should see a JSON response
3. Hit `http://your-ha-ip:5000/plot/latest` — triggers a manual plot and sends a notification
4. Pull a shot — the add-on detects it automatically and sends a notification

---

### Optional Step B — Copy the history script

Copy `gaggiuino_barista_history.py` to your HA config scripts folder:

```yaml
/config/scripts/gaggiuino_barista_history.py
```

---

### Optional Step C — Add sensors to configuration.yaml

**Command line sensor** — reads last 5 shots + graph files (add under `command_line:`):

```yaml
command_line:
  - sensor:
      name: "Gaggiuino Barista Shot History"
      unique_id: gaggiuino_barista_shot_history
      command: "python3 /config/scripts/gaggiuino_barista_history.py"
      scan_interval: 30
      json_attributes:
        - shots
        - graphs
        - total_shots
      value_template: "{{ value_json.total_shots }}"
```

Entity created: `sensor.gaggiuino_barista_shot_history`

**Template sensors** — 5 graph URL sensors for the picture grid card (add under `template:`):

```yaml
template:
  - sensor:
      - name: "Gaggiuino Barista Graph 1"
        unique_id: gaggiuino_barista_graph_1
        state: >
          {% set g = state_attr('sensor.gaggiuino_barista_shot_history', 'graphs') %}
          {{ g[0].url if g and g | length > 0 else '' }}
      - name: "Gaggiuino Barista Graph 2"
        unique_id: gaggiuino_barista_graph_2
        state: >
          {% set g = state_attr('sensor.gaggiuino_barista_shot_history', 'graphs') %}
          {{ g[1].url if g and g | length > 1 else '' }}
      - name: "Gaggiuino Barista Graph 3"
        unique_id: gaggiuino_barista_graph_3
        state: >
          {% set g = state_attr('sensor.gaggiuino_barista_shot_history', 'graphs') %}
          {{ g[2].url if g and g | length > 2 else '' }}
      - name: "Gaggiuino Barista Graph 4"
        unique_id: gaggiuino_barista_graph_4
        state: >
          {% set g = state_attr('sensor.gaggiuino_barista_shot_history', 'graphs') %}
          {{ g[3].url if g and g | length > 3 else '' }}
      - name: "Gaggiuino Barista Graph 5"
        unique_id: gaggiuino_barista_graph_5
        state: >
          {% set g = state_attr('sensor.gaggiuino_barista_shot_history', 'graphs') %}
          {{ g[4].url if g and g | length > 4 else '' }}
```

Entities created: `sensor.gaggiuino_barista_graph_1` through `_5`

---

### Optional Step D — Restart Home Assistant

After editing `configuration.yaml`, go to **Developer Tools → YAML → Restart**.

---

### Optional Step E — Install config-template-card (for graph grid)

Card 4 (graph grid) requires the `config-template-card` custom frontend card:

1. Go to **HACS → Frontend → Explore & Download**
2. Search for `config-template-card`
3. Download and restart HA

GitHub: https://github.com/iantrich/config-template-card

---

### Optional Step F — Add dashboard cards

Add cards via **Dashboard → Edit → Add Card → Manual**.

**Card 1 — Latest shot stats + AI analysis + shot graph:**

```yaml
type: vertical-stack
cards:
  - type: markdown
    title: ☕☕☕ Last Espresso Shot ☕☕☕
    content: >-
      {% set s    = state_attr('sensor.gaggiuino_barista_last_shot', 'datetime')
      %}

      {% set prof = state_attr('sensor.gaggiuino_barista_last_shot', 'profile')
      %}

      {% set dur  = state_attr('sensor.gaggiuino_barista_last_shot',
      'duration_s') %}

      {% set w    = state_attr('sensor.gaggiuino_barista_last_shot',
      'final_weight_g') %}

      {% set tw   = state_attr('sensor.gaggiuino_barista_last_shot',
      'target_weight_g') %}

      {% set p    = state_attr('sensor.gaggiuino_barista_last_shot',
      'max_pressure_bar') %}

      {% set t    = state_attr('sensor.gaggiuino_barista_last_shot',
      'water_temp_c') %}

      {% set ai   = state_attr('sensor.gaggiuino_barista_last_shot',
      'ai_available') %}

      {% set v    = state_attr('sensor.gaggiuino_barista_last_shot', 'verdict')
      %}

      {% set tips = state_attr('sensor.gaggiuino_barista_last_shot', 'tuning')
      %}

      {% set yield_str = w ~ 'g / ' ~ tw ~ 'g' if tw and tw != '-' and tw !=
      'None' else w ~ 'g' %}

      {% if prof and prof != 'None' %}

      **{{ s }}**    ☕ {{ prof }}

      ⏱  {{ dur }}s - ⚖️ {{ yield_str }}  -  📈  {{ p }} bar - 🌡️ {{ t }}°C 


      ---

      {% if ai %}🧠 **AI Analysis**


      {{ v }}

      {% if tips %}🔧 **Next shot:**

      {% for tip in tips %}- {{ tip }}

      {% endfor %}{% endif %}

      {% else %}*AI analysis unavailable*

      {% endif %}

      {% else %}*No shots recorded yet — pull a shot first!*

      {% endif %}
    card_mod:
      style: |
        ha-card {
          border-bottom-left-radius: 0 !important;
          border-bottom-right-radius: 0 !important;
          margin-bottom: -2px;text-align: center;
        }        
  - type: custom:config-template-card
    entities:
      - sensor.gaggiuino_barista_graph_1
    card:
      type: picture
      image: ${states['sensor.gaggiuino_barista_graph_1'].state}
      tap_action:
        action: url
        url_path: ${states['sensor.gaggiuino_barista_graph_1'].state}
      card_mod:
        style: |
          ha-card {
            border-top-left-radius: 0 !important;
            border-top-right-radius: 0 !important;
          }
```

**Card 2 — Shot History:**

```yaml
type: vertical-stack
cards:
  - type: markdown
    content: "## 📋 Shot History 📋"
    card_mod:
      style: |
        ha-card {
          border-bottom-left-radius: 0 !important;
          border-bottom-right-radius: 0 !important;
          margin-bottom: -2px;text-align: center;
        }
  - type: custom:config-template-card
    entities:
      - sensor.gaggiuino_barista_shot_history
      - sensor.gaggiuino_barista_graph_2
    card:
      type: vertical-stack
      cards:
        - type: markdown
          content: |-
            ${(() => {
              const shots = states['sensor.gaggiuino_barista_shot_history'].attributes.shots;
              if (!shots || shots.length < 2) return '*No data*';
              const s = shots[1];
              return '**' + s.datetime.substring(0,10) + ' ' + s.datetime.substring(11,16) + ' — ' + s.profile + '**  \n⏱ ' + s.duration_s + 's &nbsp;|&nbsp; ⚖️ ' + s.final_weight_g + 'g &nbsp;|&nbsp; 📈 ' + s.max_pressure_bar + ' bar &nbsp;|&nbsp; 🌡️ ' + s.water_temp_c + '°C';
            })()}
          card_mod:
            style: |
              ha-card {
                border-bottom-left-radius: 0 !important;
                border-bottom-right-radius: 0 !important;
                margin-bottom: -2px;text-align: center;
              }
        - type: custom:config-template-card
          entities:
            - sensor.gaggiuino_barista_graph_2
          card:
            type: picture
            image: ${states['sensor.gaggiuino_barista_graph_2'].state}
            tap_action:
              action: url
              url_path: ${states['sensor.gaggiuino_barista_graph_2'].state}
            card_mod:
              style: |
                ha-card {
                  border-top-left-radius: 0 !important;
                  border-top-right-radius: 0 !important;
                }
  - type: custom:config-template-card
    entities:
      - sensor.gaggiuino_barista_shot_history
      - sensor.gaggiuino_barista_graph_3
    card:
      type: vertical-stack
      cards:
        - type: markdown
          content: |-
            ${(() => {
              const shots = states['sensor.gaggiuino_barista_shot_history'].attributes.shots;
              if (!shots || shots.length < 3) return '*No data*';
              const s = shots[2];
              return '**' + s.datetime.substring(0,10) + ' ' + s.datetime.substring(11,16) + ' — ' + s.profile + '**  \n⏱ ' + s.duration_s + 's &nbsp;|&nbsp; ⚖️ ' + s.final_weight_g + 'g &nbsp;|&nbsp; 📈 ' + s.max_pressure_bar + ' bar &nbsp;|&nbsp; 🌡️ ' + s.water_temp_c + '°C';
            })()}
          card_mod:
            style: |
              ha-card {
                border-bottom-left-radius: 0 !important;
                border-bottom-right-radius: 0 !important;
                margin-bottom: -2px;text-align: center;
              }
        - type: custom:config-template-card
          entities:
            - sensor.gaggiuino_barista_graph_3
          card:
            type: picture
            image: ${states['sensor.gaggiuino_barista_graph_3'].state}
            tap_action:
              action: url
              url_path: ${states['sensor.gaggiuino_barista_graph_3'].state}
            card_mod:
              style: |
                ha-card {
                  border-top-left-radius: 0 !important;
                  border-top-right-radius: 0 !important;
                }
  - type: custom:config-template-card
    entities:
      - sensor.gaggiuino_barista_shot_history
      - sensor.gaggiuino_barista_graph_4
    card:
      type: vertical-stack
      cards:
        - type: markdown
          content: |-
            ${(() => {
              const shots = states['sensor.gaggiuino_barista_shot_history'].attributes.shots;
              if (!shots || shots.length < 4) return '*No data*';
              const s = shots[3];
              return '**' + s.datetime.substring(0,10) + ' ' + s.datetime.substring(11,16) + ' — ' + s.profile + '**  \n⏱ ' + s.duration_s + 's &nbsp;|&nbsp; ⚖️ ' + s.final_weight_g + 'g &nbsp;|&nbsp; 📈 ' + s.max_pressure_bar + ' bar &nbsp;|&nbsp; 🌡️ ' + s.water_temp_c + '°C';
            })()}
          card_mod:
            style: |
              ha-card {
                border-bottom-left-radius: 0 !important;
                border-bottom-right-radius: 0 !important;
                margin-bottom: -2px;text-align: center;
              }
        - type: custom:config-template-card
          entities:
            - sensor.gaggiuino_barista_graph_4
          card:
            type: picture
            image: ${states['sensor.gaggiuino_barista_graph_4'].state}
            tap_action:
              action: url
              url_path: ${states['sensor.gaggiuino_barista_graph_4'].state}
            card_mod:
              style: |
                ha-card {
                  border-top-left-radius: 0 !important;
                  border-top-right-radius: 0 !important;
                }
  - type: custom:config-template-card
    entities:
      - sensor.gaggiuino_barista_shot_history
      - sensor.gaggiuino_barista_graph_5
    card:
      type: vertical-stack
      cards:
        - type: markdown
          content: |-
            ${(() => {
              const shots = states['sensor.gaggiuino_barista_shot_history'].attributes.shots;
              if (!shots || shots.length < 5) return '*No data*';
              const s = shots[4];
              return '**' + s.datetime.substring(0,10) + ' ' + s.datetime.substring(11,16) + ' — ' + s.profile + '**  \n⏱ ' + s.duration_s + 's &nbsp;|&nbsp; ⚖️ ' + s.final_weight_g + 'g &nbsp;|&nbsp; 📈 ' + s.max_pressure_bar + ' bar &nbsp;|&nbsp; 🌡️ ' + s.water_temp_c + '°C';
            })()}
          card_mod:
            style: |
              ha-card {
                border-bottom-left-radius: 0 !important;
                border-bottom-right-radius: 0 !important;
                margin-bottom: -2px;text-align: center;
              }
        - type: custom:config-template-card
          entities:
            - sensor.gaggiuino_barista_graph_5
          card:
            type: picture
            image: ${states['sensor.gaggiuino_barista_graph_5'].state}
            tap_action:
              action: url
              url_path: ${states['sensor.gaggiuino_barista_graph_5'].state}
            card_mod:
              style: |
                ha-card {
                  border-top-left-radius: 0 !important;
                  border-top-right-radius: 0 !important;
                }
```

---

## Add-on configuration reference

| Option | Required | Description | Example |
|--------|----------|-------------|---------|
| `api_base` | ✅ Yes | Gaggiuino IP or hostname | `http://gaggiuino.local` |
| `anthropic_api_key` | Recommended | Anthropic API key (reliable, ~$0.001/shot) | `sk-ant-...` |
| `gemini_api_key` | ✅ Yes | Google Gemini API key | `AIzaSy...` |
| `ha_notify_service` | ✅ Yes | HA notify service for your phone | `notify.mobile_app_johns_iphone` |

---

## API Endpoints

Port **5000** is exposed by the add-on.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/plot/latest` | GET / POST | Manually trigger plot + notification |
| `/status` | GET | Watcher state + live machine data |

Access at `http://your-ha-ip:5000/status`

### /status response example

```json
{
  "watcher_status": "idle",
  "shot_running": false,
  "shot_elapsed_s": null,
  "known_shot_id": 64,
  "last_plot": "2026-03-21T20:31:12",
  "last_error": null,
  "machine": {
    "pressure": 0.11,
    "brew_switch": false,
    "temperature": 93.0,
    "weight": 0.0,
    "profile": "Leva 9 v0.9 (community)"
  }
}
```

---

## Shot graph contents

| Element | Axis | Color |
|---------|------|-------|
| Temperature | Left (°C) | Orange |
| Target temperature | Left (°C) | Orange dashed |
| Pressure | Right (bar) | Blue |
| Target pressure | Right (bar) | Blue dashed |
| Pump flow | Right (ml/s) | Yellow |
| Target pump flow | Right (ml/s) | Yellow dashed |
| Weight flow | Right (ml/s) | Green |
| Shot weight | Hidden (g) | Purple |

Graphs saved to `/homeassistant/www/gaggiuino-barista/` — up to 30 kept automatically.
Last shot always available at `/local/gaggiuino-barista/last_shot.png`

---

## Shot detection parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Poll interval | 3s | Machine status check frequency |
| Post-shot delay | 8s | Wait after shot saved before fetching |
| Min shot duration | 8s | Shorter events ignored |
| Max shot duration | 180s | Longer events ignored |
| Offline poll | 30s | Interval when machine unreachable |
| Gemini min interval | 70s | Rate limit protection |

---

## Troubleshooting

**No notification received**
- Verify `ha_notify_service` matches exactly what appears in Developer Tools → Actions
- Confirm `homeassistant_api: true` is in `config.yaml` and you clicked **Rebuild**
- Test: Developer Tools → Actions → your notify service → `{"message": "test"}`

**Cannot access `/status` or `/plot/latest`**
- Confirm `ports: 5000/tcp: 5000` is in `config.yaml` and you clicked **Rebuild**
- Access via `http://your-ha-ip:5000/status`

**Card 1 shows "No shots recorded yet" even after a shot**
- The REST sensor URL must use your actual HA IP, not `localhost`
- Check `configuration.yaml` → `rest:` → `resource:` — replace `localhost` with your HA IP
- Trigger a manual plot at `http://your-ha-ip:5000/plot/latest` to generate `last_shot.json`

**Status shows `offline` with machine on**
- Verify `api_base` — open `http://<ip>/api/system/status` in your browser

**Shot not detected automatically**
- Check the Log tab for `Shot STARTED` messages
- Use `/plot/latest` to manually test

**AI annotations missing from graph**
- Gemini rate limit or daily quota — graph and notification still work
- Daily quota resets at midnight Pacific Time (08:00 Greek time)
- Check logs for `Gemini daily quota exceeded` or `Gemini per-minute rate limit hit`

**Card graphs not showing**
- Install `config-template-card` from HACS → Frontend
- Verify `sensor.gaggiuino_barista_graph_1` has a state in Developer Tools → States
- Pull a shot and wait 30s for the command_line sensor to refresh