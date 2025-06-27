# Isticmaal Python 3.10 base image
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose port (waa in la jaanqaadaa port-ka aad Start Command ku isticmaaleyso)
EXPOSE 10000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"] 