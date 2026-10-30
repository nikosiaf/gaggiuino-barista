#!/usr/bin/with-contenv bashio
#
# Gaggiuino Barista - Startup Script
#
# This script is executed when the Docker container starts.
# It reads configuration from Home Assistant and exports environment variables
# before launching the Python server.

echo "Starting Gaggiuino-Barista ..."

# Read Gaggiuino API base URL from add-on configuration
API_BASE=$(bashio::config 'api_base')
export API_BASE

# Read Anthropic API key for Claude Haiku (primary AI provider)
ANTHROPIC_API_KEY=$(bashio::config 'anthropic_api_key')
export ANTHROPIC_API_KEY

# Read Google Gemini API key (fallback AI provider)
GEMINI_API_KEY=$(bashio::config 'gemini_api_key')
export GEMINI_API_KEY

# Token that POST /plot/latest requires (empty: the endpoint stays off)
API_TOKEN=$(bashio::config 'api_token')
export API_TOKEN

# Optional AI model overrides
if bashio::config.has_value 'anthropic_model'; then
    ANTHROPIC_MODEL=$(bashio::config 'anthropic_model')
    export ANTHROPIC_MODEL
fi
if bashio::config.has_value 'gemini_model'; then
    GEMINI_MODEL=$(bashio::config 'gemini_model')
    export GEMINI_MODEL
fi

# Read Home Assistant notify service for mobile notifications
HA_NOTIFY_SERVICE=$(bashio::config 'ha_notify_service')
export HA_NOTIFY_SERVICE

# Read language setting for AI analysis output (en, el, it, de, es, fr)
LLM_LANGUAGE=$(bashio::config 'llm_language')
export LLM_LANGUAGE

# Launch the Python server
exec python3 /app/src/server.py
