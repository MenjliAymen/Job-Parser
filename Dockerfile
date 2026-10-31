FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY Job_parser ./Job_parser

# Ollama runs on the host by default; override OLLAMA_URL for other setups.
ENV OLLAMA_URL=http://host.docker.internal:11434 \
    DATA_DIR=/app/data
VOLUME /app/data
EXPOSE 8000

CMD ["uvicorn", "Job_parser.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
