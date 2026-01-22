FROM python:3.14

# Set working directory inside container
WORKDIR /app

# Copy all project files into container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command (adjust if using a script)
CMD ["python", "app.py"]