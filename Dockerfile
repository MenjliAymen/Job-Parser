FROM python:3.10-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

CMD ["uvicorn", "Job_parser.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
