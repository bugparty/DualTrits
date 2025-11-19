FROM ubuntu:25.04

# Install Dependencies
RUN apt-get update && \
    apt-get install -y build-essential cmake \
    libmpfr-dev libmpfrc++-dev \
    libgmp-dev \
    git \
    && \
    rm -rf /var/lib/apt/lists/*

# Copy Project
WORKDIR /DualTrits
COPY . .

# Build Project
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release

CMD ["./build/project_float"]
