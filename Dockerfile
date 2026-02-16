# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /code

# Copy requirements and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the application folders
COPY ./app /code/app
COPY ./frontend /code/frontend
COPY ./data /code/data

# Grant write permissions to the app folder (for the CSV logs to work)
RUN chmod -R 777 /code/app

# Expose the ports (7860 is for Hugging Face to show the UI)
EXPOSE 7860
EXPOSE 8000

# Start both Uvicorn (API) and Streamlit (Frontend)
# Uvicorn runs in the background (&) on port 8000
# Streamlit runs in the foreground on port 7860
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run frontend/dashboard.py --server.port 7860 --server.address 0.0.0.0