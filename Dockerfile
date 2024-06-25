FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    pkg-config \
    libglib2.0-dev \
    libhdf5-dev \
    build-essential

COPY requirements.txt /app/requirements.txt

# Install h5py with no-binary flag
RUN pip install --no-binary h5py h5py

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
