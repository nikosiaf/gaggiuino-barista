ARG BUILD_FROM
FROM ${BUILD_FROM}

RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-matplotlib \
    py3-requests \
    py3-flask

RUN pip3 install --break-system-packages waitress 2>/dev/null || \
    pip3 install waitress 2>/dev/null || \
    echo "waitress install failed - will use Flask dev server"

RUN mkdir -p /app/src

WORKDIR /app

COPY run.sh /run.sh
COPY src/server.py /app/src/
COPY src/plot_logic.py /app/src/
COPY src/annotation_engine.py /app/src/

RUN chmod a+x /run.sh

CMD ["/run.sh"]
