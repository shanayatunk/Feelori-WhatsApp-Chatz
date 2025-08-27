# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# --- THIS IS THE FIX ---
# Command to run the application with the correct module path.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]