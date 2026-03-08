# STEP 1: Choose a Base Image
# We use python:3.10-slim to keep the image size small (~200MB instead of 1GB+)
FROM python:3.10-slim

# STEP 2: Set Environment Variables
# Prevents Python from writing .pyc files and ensures output is sent to logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# STEP 3: Install System Dependencies
# 'build-essential' is needed for numpy/scikit-learn math optimizations
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# STEP 4: Set the Working Directory
WORKDIR /app

# STEP 5: Install Python Dependencies
# We copy requirements.txt first to leverage Docker's cache layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STEP 6: Copy your Application Code
COPY . .

# STEP 7: Expose the Port
EXPOSE 8000

# STEP 8: The Startup Command
# '0.0.0.0' allows the container to be reached from your host machine
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]