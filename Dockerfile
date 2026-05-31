# Use the official Python base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first for better layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port on which the server will run
EXPOSE 8000

# Run as non-root user for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# Start the FastAPI server with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
