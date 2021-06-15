FROM python:3.8
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY step4/train.py train.py
COPY step4/server.py server.py
ENTRYPOINT uvicorn server:app --host 0.0.0.0