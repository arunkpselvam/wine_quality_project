# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY app/requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY app/ /app
COPY model/ /app/model

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app/main.py when the container launches
CMD ["python", "main.py"]