FROM python:3.9-slim

# Install build essentials and other dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install numpy first
RUN pip install numpy

# Copy requirements and setup files
COPY setup.py ./
COPY MANIFEST.in ./
COPY README.md ./

# Copy source code and lib package
COPY core ./core/
COPY lib ./lib/ 

# Build the shared object
RUN python setup.py build_ext --inplace

# Ensure the .so file is renamed to cpu_chess.so and moved to /app/lib
#RUN mkdir -p /app/lib \
#    && cp build/lib.linux-x86_64-3.9/lib/cpu_chess*.so /app/lib/ \
#    && mv /app/lib/cpu_chess*.so /app/lib/cpu_chess.so

# Default command
CMD ["python3"]
