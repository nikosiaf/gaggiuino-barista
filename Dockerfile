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

WORKDIR /app

COPY run.sh /run.sh
COPY server.py /app/
COPY plot_logic.py /app/
COPY annotation_engine.py /app/

RUN chmod a+x /run.sh

CMD ["/run.sh"]