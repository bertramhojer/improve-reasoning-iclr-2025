FROM python:3.11-slim

# Install system dependencies 
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml requirements.txt ./

# Install dependencies directly with pip
RUN pip install uv
RUN uv venv --python 3.11
RUN uv pip install -r requirements.txt

# Copy the rest of your code
COPY src /app/src