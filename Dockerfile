# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies (for Pillow image support)
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and image
COPY app.py /app/
COPY volleyball.py /app/
COPY target_window.py /app/
COPY Volleyball_Shoot_Processed.png /app/

# Expose port (default Dash port)
EXPOSE 8050

# Run the app
CMD ["python", "app.py"]
