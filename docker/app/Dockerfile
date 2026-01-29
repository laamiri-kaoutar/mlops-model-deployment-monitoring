# Use python slim image
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y libgomp1

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
# We copy 'src' into '/app/src'
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Set Python path so it finds 'src'
ENV PYTHONPATH=/app

# Command to run the app
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]