# Use the official Python 3.11 base image
FROM python:3.11-slim

# Install dependencies for dlib, including CMake
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

## Make port 80 available to the world outside this container
#EXPOSE 80

## Define environment variable
#ENV NAME World
#
## Run app.py when the container launches
#CMD ["python", "app.py"]

# FOR FIREBASE STUFF

ENV PORT 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8  app:app
