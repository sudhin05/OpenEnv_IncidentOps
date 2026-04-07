FROM python:3.11-slim

WORKDIR /server

COPY . /server

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
