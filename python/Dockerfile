FROM python:3.10-slim
COPY req.txt .
RUN pip3 install -r req.txt
COPY . .
# CMD ["gunicorn", "-w","3", "--bind", "0.0.0.0:5000", "app:app"]