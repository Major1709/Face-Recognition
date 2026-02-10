FROM python:3.10.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -U pip && pip install -r requirements.txt

COPY app /app/app

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
