FROM python:3.10-slim

WORKDIR /usr/src/app

COPY app/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

CMD ["python", "./main.py"]