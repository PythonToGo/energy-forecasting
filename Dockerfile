FROM python:3.10-slim

# Working directory
WORKDIR /app

# Copy requirements.
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Port
EXPOSE 8000
EXPOSE 8502
EXPOSE 5050

# FastAPI, Streamlit, MLflow
CMD ["bash", "run_all.sh"]

