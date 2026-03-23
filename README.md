<p align="center">
  <img src="https://raw.githubusercontent.com/nikosiaf/gaggiuino-barista/main/logo.png" alt="Gaggiuino Barista Logo" width="220">
</p>

# Gaggiuino Barista - Home Assistant Add-on for Gaggiuino modified espresso machines.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/nikosiaf/gaggiuino-barista/blob/main/LICENSE)

[Home Assistant](https://www.home-assistant.io/) add-on for [Gaggiuino](https://github.com/Zer0-bit/gaggiuino) espresso machines.

Automatically detects shots, generates detailed graphs, runs a deterministic telemetry analyzer plus AI phrasing, and sends mobile push notifications with annotated graphs and next-shot tuning recommendations.

## What's new in 1.1.0

- New **hybrid annotation engine**
  - deterministic telemetry analyzer computes extraction features
  - deterministic event detector finds anchored moments in the shot
  - Anthropic rewrites those events into clean graph labels, verdict, tuning, and score
  - Gemini fallback uses the **same JSON schema and prompt contract**
- New AI output fields in `last_shot.json`
  - `ai_provider`
  - `score`
  - `confidence`
  - `features`
  - `detected_events`
- More stable AI behavior: the LLM no longer invents the shot structure from scratch

## Features

- Auto shot detection via brew switch + shot ID confirmation
- Detailed shot graph: pressure, flow, temperature, weight curves with glow effects
- Deterministic telemetry analyzer for extraction features and event detection
- AI-powered graph annotations with timestamp anchors
- Anthropic primary provider, Gemini fallback
- Shot score (`0–100`), verdict, and tuning recommendations
- Mobile push notification with graph image
- JSON shot data written to `/local/gaggiuino-barista/last_shot.json` for HA template sensors
- Manual trigger and health check HTTP endpoints
- Graceful offline handling when machine is unreachable

## Installation

1. In Home Assistant go to **Settings -> Add-ons -> Add-on Store**
2. Click the three dots menu (top right) -> **Repositories**
3. Add: `https://github.com/nikosiaf/gaggiuino-barista`
4. Find **Gaggiuino Barista** in the store and install it

## Quick Setup

1. Add your Gaggiuino base URL and notify service
2. Add an **Anthropic API key** for primary AI phrasing
3. Optionally add a **Gemini API key** for fallback
4. Add the REST sensor from `gaggiuino_barista/gaggiuino_barista_sensors.yaml` to your HA config
5. Start or rebuild the add-on

See the [Documentation](https://github.com/nikosiaf/gaggiuino-barista/blob/main/docs/DOCS.md) for full setup instructions.

## AI architecture

The add-on now uses this pipeline:

```text
shot telemetry
  -> deterministic feature extraction
  -> deterministic event detection
  -> LLM phrasing layer
  -> graph overlay + JSON + mobile notification
```

This keeps the extraction logic stable while still giving readable annotations.

## Credits

- [Gaggiuino](https://github.com/Zer0-bit/gaggiuino) - Open source espresso machine controller.
- [Home Assistant](https://www.home-assistant.io/) - Open source home automation that puts local control and privacy first.
- [Config Template Card Card](https://github.com/iantrich/config-template-card) - This card is for Lovelace on Home Assistant that allows you to use pretty much any valid Javascript on the hass object in your configuration.