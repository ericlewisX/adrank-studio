FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

COPY requirements/requirements-lock.txt .

RUN pip install -r requirements-lock.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]