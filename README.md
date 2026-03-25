<p align="center">
  <img src="https://raw.githubusercontent.com/nikosiaf/gaggiuino-barista/main/logo.png" alt="Gaggiuino Barista Logo" width="280">
</p>

<h1 align="center">☕ Gaggiuino Barista</h1>

<p align="center">
  <a href="https://github.com/nikosiaf/gaggiuino-barista/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Personal%20Use-red" alt="License"></a>
  <a href="https://www.home-assistant.io/"><img src="https://img.shields.io/badge/Home%20Assistant-Add--on-blue" alt="HA"></a>
  <img src="https://img.shields.io/badge/Version-2.0.1-green" alt="v2.0.1">
</p>

<p align="center">
  <strong>AI-powered espresso shot analysis for Home Assistant</strong><br>
  Auto-detect shots • Beautiful graphs • Smart feedback • Mobile notifications
</p>

---

## ✨ What's New in 2.0.0

| Feature | Description |
|---------|-------------|
| 🎯 **Profile-Aware Analysis** | Matches your shots against community or custom profiles |
| 🌍 **Multi-Language** | AI feedback in English, Greek, Italian, German, Spanish, French |
| 📊 **Flow Ratio Tracking** | Detects channeling and restriction issues |
| 🧠 **Taste Prediction** | Predicts extraction quality (well_extracted → poorly_extracted) |
| ⭐ **Smarter Scoring** | Calibrated to match human evaluation |

---

## 🚀 Quick Start

```bash
# 1. Install from HA Add-on Store → Add Repository → https://github.com/nikosiaf/gaggiuino-barista

# 2. Configure in Add-on UI:
   • api_base: http://gaggiuino.local
   • anthropic_api_key: sk-ant-... (recommended, ~$0.001/shot)
   • gemini_api_key: (optional fallback)
   • ha_notify_service: notify.mobile_app_your_phone
   • llm_language: en (or el, it, de, es, fr)

# 3. Add REST sensor to configuration.yaml:
rest:
  - resource: "http://YOUR-HA-IP:8123/local/gaggiuino-barista/last_shot.json"
    scan_interval: 10
    sensor:
      - name: "Gaggiuino Last Shot"
        unique_id: gaggiuino_barista_last_shot
        value_template: "{{ value_json.shot_id }}"
        json_attributes: "{{ value_json.keys() | list }}"
```

---

## 📱 What You Get

```
☕ Shot Score: 85/100 ☕
🌡93°C 📈9.4bar ⚖48.6g ⏱33s

🔧 Grind finer to slow the opening
```

**Plus:**
- 📈 Detailed shot graphs with pressure/flow/temperature curves
- 🏷️ AI annotations at key extraction moments
- 🔔 Push notification to your phone
- 📊 Shot history (last 10 shots)

---

## 🧠 How It Works

```
Shot Data → Deterministic Analysis → LLM Phrasing → Graph + Notification
     ↓              ↓                    ↓              ↓
 Gaggiuino    Feature Extraction     (Claude/Gemini)  HA Push
 API          Event Detection        Natural Language  + Graph
```

**Analysis includes:**
- First drops timing
- Pre-infusion quality (wetting → soaking → compression)
- Pressure/flow stability
- Flow ratio (channeling detection)
- Profile adherence scoring
- Taste classification

---

## 🎯 Score Guide

| Score | Rating | What it means |
|-------|--------|---------------|
| 80+ | ⭐ Excellent | Hit targets, stable extraction |
| 70-79 | 👍 Good | Minor issues, still pleasant |
| 60-69 | 😐 Drinkable | Noticeable problems |
| <60 | ⚠️ Fix it | Significant issues detected |

---

## 📦 Credits

- [Gaggiuino](https://github.com/Zer0-bit/gaggiuino) - Community espresso profiles
- [Home Assistant](https://www.home-assistant.io/) - Home automation platform
- [Community Profiles](https://github.com/Zer0-bit/gaggiuino/tree/community/profiles) - CC BY-NC 4.0

---

<p align="center">
  <a href="https://github.com/nikosiaf/gaggiuino-barista/issues">🐛 Issues</a> •
  <a href="https://github.com/nikosiaf/gaggiuino-barista/discussions">💬 Discussions</a> •
  <a href="https://github.com/nikosiaf/gaggiuino-barista/tree/main/docs">📖 Full Docs</a>
</p>
