FROM python:3.10-slim
COPY requirements.txt .
CMD pip3 install -r requirements.txt
COPY . .