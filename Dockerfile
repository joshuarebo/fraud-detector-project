# Use an official Python 3.9+ image (Debian-based slim for efficiency)
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy everything from the local project directory (including models/)
COPY . .  

# Ensure models and logs directories exist
RUN mkdir -p /app/models /app/logs

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*  

# Upgrade pip and install required dependencies
RUN pip install --upgrade pip setuptools wheel

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the API port (Cloud Run defaults to 8080)
ENV PORT=8080
EXPOSE 8080

# Start the API
CMD ["python", "fraud_api.py"]
