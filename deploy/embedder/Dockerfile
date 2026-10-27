# Separate from the API image so torch is not installed into ragkit itself.
# The first start downloads BAAI/bge-m3 into the cache volume.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    "sentence-transformers==6.1.0" \
    "fastapi>=0.115" \
    "uvicorn>=0.30"
COPY server.py .
EXPOSE 80
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
