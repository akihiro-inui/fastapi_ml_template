# Base Python image
FROM python:3.8

# Port to expose
EXPOSE 80

# Copy contents
COPY .. .

# Install libraries
RUN apt-get update

# Install python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run API application
CMD ["python", "src/main.py"]