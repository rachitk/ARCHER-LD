# Base image: micromambda with CUDA
FROM mambaorg/micromamba:2.9.0-cuda13.2.1-ubuntu24.04

USER root

# Install necessary dependencies for OpenMPI (network and others)
RUN apt-get update && apt-get install -y \
    gocryptfs \
    libucx-dev \
    libfabric-dev \
    libslurm-dev \
    libhwloc-dev \
    libnuma-dev \
    libibverbs-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install OpenMPI (TODO: compile from source instead of using apt-get to get newer versions?)
RUN apt-get update && apt-get install -y openmpi-bin libopenmpi-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# TODO: offer build option where people can build OpenMPI themselves from source
# (maybe also allow the to specify the OpenMPI version to match the host)

# Note that the default workdir for this base image is /tmp, but we explicitly define it to avoid issues
# (specifically, we work in /home/mambauser, which is the home directory for the mambauser user in this base image)
WORKDIR /home/mambauser

# Clone the repository with the environment and code for LD-GPU
# Install git using apt-get
# RUN apt-get update && apt-get install -y git && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*
# ARG repo_url="https://github.com/rachitk/ARCHER-LD.git"
# RUN git clone $repo_url

# If running directly from a cloned repo already
RUN mkdir -p ARCHER-LD
COPY environment.yml ARCHER-LD
COPY ld_gpu.py ARCHER-LD
COPY kernel_functions.py ARCHER-LD
COPY ld_functions.py ARCHER-LD
COPY utils.py ARCHER-LD
COPY convert_utils ARCHER-LD/convert_utils

# Remove OpenMPI from the environment.yml file to avoid conflicts with the system OpenMPI
RUN sed -i "/- openmpi/d" ARCHER-LD/environment.yml

# Add disabling of timeout for Mamba (packages can be quite large, especially cuda)
ENV MAMBA_NO_LOW_SPEED_LIMIT=1

# Set up the environment using micromamba
RUN micromamba install -y -n base -f ARCHER-LD/environment.yml && \
    micromamba clean --all --yes

# Set user back to mambauser for running the application 
# (this is the default user in the base image)
USER mambauser

# Entry point for the container, which will run the Python script with any arguments passed to the container 
# (run ld_gpu.py with all the arguments the user passes)
# Note, we explicitly specify the path since Apptainer can change the working directory when running the container
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "/home/mambauser/ARCHER-LD/ld_gpu.py"]