FROM python:3.8.4-slim
WORKDIR /proj
COPY requirements.txt .
RUN \
    pip install -r ./requirements.txt
COPY . .
# CMD ["python", "app.py"]
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8005"]
