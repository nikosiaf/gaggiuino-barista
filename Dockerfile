# Gaggiuino Barista - Docker Image Definition
#
# Base image is provided by Home Assistant at build time
ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install system dependencies (Alpine Linux packages)
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-matplotlib \
    py3-requests \
    py3-flask \
    py3-waitress

# Create application directories
RUN mkdir -p /app/src /app/profiles

WORKDIR /app

# Copy application files into the container
COPY run.sh /run.sh
COPY src/server.py /app/src/
COPY src/plot_logic.py /app/src/
COPY src/annotation_engine.py /app/src/
COPY src/redact.py /app/src/

# Bundle 44 community profiles for automatic profile matching
# Licensed under CC BY-NC 4.0 (see profiles/ directory for attribution)
COPY profiles/ /app/profiles/

# Make startup script executable
RUN chmod a+x /run.sh

# Start the add-on
CMD ["/run.sh"]
