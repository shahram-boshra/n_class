# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install RDKit dependencies (if needed, adjust to your system)
# Example for Debian-based systems:
RUN apt-get update && apt-get install -y \
    libstdc++6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Assuming you have a requirements.txt file with all your Python dependencies
# If not, you'll need to install them manually here, like:
# RUN pip install torch torch_geometric rdkit-pypi pandas scikit-learn matplotlib pyyaml diskcache pydantic joblib

# Set environment variables if needed
# ENV MY_VAR=my_value

# Expose any ports needed by your application (if any)
# EXPOSE 8080

# Define environment variable
ENV PYTHONPATH /app

# Run app.py when the container launches. If your main script is different, change it.
CMD ["python", "app.py"] # or the name of your main python file.