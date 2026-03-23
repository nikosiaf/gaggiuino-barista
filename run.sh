#!/usr/bin/with-contenv bashio

echo "Starting Gaggiuino-Barista ..."

API_BASE=$(bashio::config 'api_base')
export API_BASE

ANTHROPIC_API_KEY=$(bashio::config 'anthropic_api_key')
export ANTHROPIC_API_KEY

GEMINI_API_KEY=$(bashio::config 'gemini_api_key')
export GEMINI_API_KEY

HA_NOTIFY_SERVICE=$(bashio::config 'ha_notify_service')
export HA_NOTIFY_SERVICE

# SUPERVISOR_TOKEN is auto-injected by HA — explicitly re-export to ensure
# subprocesses (plot_logic.py) inherit it
export SUPERVISOR_TOKEN

python /app/src/server.py