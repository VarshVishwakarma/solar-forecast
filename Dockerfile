FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r app/requirements.txt

EXPOSE 7860
EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/dashboard.py --server.port 7860 --server.address 0.0.0.0
