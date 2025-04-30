# Use an official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean

# Copy the requirements file into the container
COPY ./ /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# TODO: attach current directory as volume in docker container to access data folde

