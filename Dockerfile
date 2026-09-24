FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.14.0 \
    torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app/ app/
COPY src/ src/
COPY models/ models/

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]