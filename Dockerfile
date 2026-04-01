FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir .
RUN startup-churn train

EXPOSE 8000

CMD ["startup-churn", "serve", "--host", "0.0.0.0", "--port", "8000"]
