# Dockerfile

FROM python:3.7.4-slim

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

COPY . /app

ENTRYPOINT ["streamlit"]
