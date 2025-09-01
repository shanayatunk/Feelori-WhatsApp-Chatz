# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables for production for better performance and security
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gunicorn, the production-ready web server
RUN pip install gunicorn

# Copy the rest of the application code
COPY . .

# Command to run the application with gunicorn for production.
# It uses 4 workers to handle multiple requests simultaneously.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]

