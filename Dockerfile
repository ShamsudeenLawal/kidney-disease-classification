# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy source folder
COPY . .

# Install requirements
RUN pip install -r requirements.txt

# Start app
CMD ["python3", "/app/app.py"]
