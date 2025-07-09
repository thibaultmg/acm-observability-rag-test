# Dockerfile

# Use a slim Python base image for a smaller final image size
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && \
  apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
# Install uv, the fast Python package installer
RUN pip install uv

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install Python dependencies using uv
# Using --system to install in the main environment as we are in a container
RUN uv pip install --no-cache --system -r requirements.txt

# Copy all your application files, including the pre-generated storage
# This "bakes in" the index and documents into the image
COPY main.py .
# COPY config.json .
COPY system_prompt.txt .
COPY product_description.txt .
# COPY ./storage/ ./storage/
COPY ./data/ ./data/

# This is necessary for OpenShift's random user security model.
# It allows Chainlit to create its '.files' directory at runtime.
RUN chmod -R 777 /app

# Expose the port the FastAPI app runs on
EXPOSE 8000

# Define the command to run your application
# The server will start when the container launches
CMD ["chainlit", "run", "main.py", "--host", "0.0.0.0", "--port", "8000", "-w"]
