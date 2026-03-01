FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt
RUN pip uninstall -y multipart || true
RUN python src/train.py

CMD ["uvicorn","src.predict:app","--host","0.0.0.0","--port","8000"]