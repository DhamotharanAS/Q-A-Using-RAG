# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Expose the port Streamlit will run on
EXPOSE 8501

# Run the FastAPI and Streamlit applications
CMD ["sh", "-c", "uvicorn streamlit:app --host 0.0.0.0 --port 8000 & streamlit run frontend.py"]