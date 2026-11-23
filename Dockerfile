FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY spamfilter ./spamfilter
COPY static ./static
COPY api.py train.py ./
# Mount these as volumes so feedback and new model versions survive container restarts:
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -e API_KEY=... -p 8000:8000 spam-service
ENV MODEL_DIR=models/v1
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]