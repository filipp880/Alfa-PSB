FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY assistant.py corpus.py proxy.py solution.py web.py requirements.txt ./
COPY static/ ./static/
COPY data/knowledge-base.zip ./data/knowledge-base.zip
USER 10001:10001
EXPOSE 8000
CMD ["python", "web.py", "--host", "0.0.0.0", "--port", "8000"]
