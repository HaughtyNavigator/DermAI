# Use an official lightweight Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# (libgl1 and libglib2.0 are often required by image processing libraries like OpenCV/Pillow under the hood)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies.
# We explicitly use the PyTorch CPU index URL here to prevent Docker from downloading the massive 4GB+ GPU version of torch,
# keeping the image extremely small and fast.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Gradio runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "frontend.py"]
