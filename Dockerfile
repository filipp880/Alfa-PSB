FROM python:3.12-slim

ARG PSB_INSTALL_TELEMETRY=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && if [ "$PSB_INSTALL_TELEMETRY" = "1" ]; then python -m pip install --no-cache-dir -r requirements-dev.txt; fi
COPY assistant.py corpus.py fact_checks.py proxy.py solution.py telemetry.py web.py ./
COPY static/ ./static/
COPY data/knowledge-base.zip ./data/knowledge-base.zip
RUN python -c "from pathlib import Path; from corpus import INDEX_FILENAME, build_prebuilt_index; data = Path('data'); (data / INDEX_FILENAME).write_bytes(build_prebuilt_index(data / 'knowledge-base.zip'))"
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=3s --start-period=70s --retries=3 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/healthz', timeout=2).close()"]
CMD ["python", "web.py", "--host", "0.0.0.0", "--port", "8000"]
