# ARCHER-LD

## Introduction

ARCHER-LD is a tool that uses GPUs and MPI to parallelize the calculation of genome-wide linkage disequilibrium (LD) in biobank-scale datasets.

[Linkage disequilibrium (LD)](https://en.wikipedia.org/wiki/Linkage_disequilibrium) is a measure of how associated two genetic variants (or alleles) are in a population. It is a pairwise function of two variants and can be computed for any pair of variants that exists within a population, and it is valuable for a wide range of downstream biomedical and population genomics analyses.

However, in large datasets, particularly at the biobank scale, there can be over 10,000,000 or even 100,000,000 variants across the entire genome, which makes computing LD in these datasets particularly challenging from a computational perspective. This has led to many limitations in how LD can be used for downstream analysis despite its broad utility.

Our goal was to make this process computationally and temporally feasible by using GPUs to accelerate the computation process. We did so by reformulating the computations involved in determining LD as a series of linear algebra operations that GPUs are particularly good at performing and then parallelizing these computations in an [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) fashion.


## How ARCHER-LD works

The primary contribution of ARCHER-LD is coordinating calculations across an arbitrary number of GPUs in a way that tries to make full use of the GPUs (both in terms of compute and VRAM). 

To do so, ARCHER-LD loads genomics data in chunks and computes LD between the variants in these chunks across the entire population. To facilitate this, ARCHER-LD also offers utilities for parallel conversion of VCF and Plink binary (bed/bim/fam) files to the [Zarr format](https://zarr.dev/), which allows for quick, chunked loading from an on-disk array. 

Various parts of ARCHER-LD are parallelized, as described below, and this parallelization is done using the [Message Passing Interface](https://en.wikipedia.org/wiki/Message_Passing_Interface) to try to maximize its portability across many different host systems.

ARCHER-LD's overall pipeline is broken up into three steps, known as Step 0, Step 1, and Step 2.

In Step 0, ARCHER-LD converts standard genomics formats into an on-disk Zarr array of size N x K (where N is the number of variants and K is the number of samples) with associated metadata about each variant. This does not benefit from GPU acceleration, and so the operation is done entirely on the CPU. However, it can be parallelized and so multiple processes are still used here, if provided.

In Step 1, ARCHER-LD iterates over chunks and computes LD by reformulating it as a set of linear algebra operations and in-place operations that can be done much more quickly on the GPU. This is the primary value of ARCHER-LD and where the actual LD computation takes place, as blocks of the LD matrix are computed in a per-GPU fashion. Note that ARCHER-LD needs one process per GPU *and* an additional process that coordinates assigning chunks to GPUs as they become available. In this manner, ARCHER-LD can operate with any number of GPUs (as chunks will be consumed by GPUs as they become available) and can be restarted if terminated before completion (by skipping already-computed blocks of the LD matrix). Each block of the LD matrix is saved to disk as it is completed. 

In Step 2, ARCHER-LD takes the computed, thresholded chunks and concatenates them into a single large sparse array, as well as storing the metadata for each variant in an easily accessible and filterable form. This is done entirely on the CPU and can take a variable amount of time, depending on how sparse or dense the LD matrix is and how many blocks were computed. This is done on a single process, as there is no benefit to using multiple processes for this purpose.


## Setup

We offer both a Conda environment and a Docker container (available on DockerHub) for running ARCHER-LD. We generally recommend using the Conda environment over the Docker container to avoid issues that might arise from MPI incompatibilities, but we do provide the container in case setting up a Conda environment is not possible for any reason.


### Conda Environment

The best way to run ARCHER-LD is to run it "bare-metal" - that is, within a system directly with dependencies installed, usually within a virtual environment to avoid conflicts. We provide an `environment.yml` that has all dependencies listed, including Python, OpenMPI, and CuPy, among others.

You should start by installing your preferred flavor of Conda. As of the time of release, we recommend [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), and our commands reflect this (but you can replace `micromamba` below with `conda` or your preferred Conda distribution). 

After setting up a Conda distribution, you will need to clone the repository to a local drectory and set up an environment by running the following commands:

```
git clone https://github.com/rachitk/ARCHER-LD.git
micromamba create --file environment.yml
```

This will install all of the necessary dependencies into an environment called `archer-ld`, which you can then activate by running:

```
micromamba activate archer-ld
```


### Docker Container

For people who are having difficulties setting up a Conda environment or operating in an enrivonment where this is not possible, we have made available a [Docker container](https://hub.docker.com/r/kumarrachit/archer-ld) with all of the necessary dependencies to run ARCHER-LD already installed.

You can pull the container using Docker (this command pulls the container to your local Docker install):

```
docker pull kumarrachit/archer-ld
```

You can also pull this container with Apptainer (this command saves the image to a file called `archer-ld.sif`): 

```
apptainer pull archer-ld.sif docker://kumarrachit/archer-ld:latest
```


## Running ARCHER-LD

### Initial note: Running ARCHER-LD with MPI (using mpirun/mpiexec)

The primary entrypoint for all ARCHER-LD steps is through `ld_gpu.py`, and most of the relevant code can be found in this file. The other files in the repository generally provide helper or utility functions (with the exception of `kernel_functions.py` which is where the actual LD calculation functions are implemented).

If you are using the provided Conda environment, running ARCHER-LD (which is a Python program) is fairly simple, but it involves using multiple processes. Our code already orchestrates the communication between processes itself. 

To launch ARCHER-LD with multiple processes, you need to use `mpirun` or `mpiexec` to launch multiple processes that can communicate with each other using MPI:

```
mpirun -n [num_processes] python ld_gpu.py [ARGS]
```

where `[ARGS]` represents the arguments described below.


If you are using the Docker container, the container is set up with an entry point such that `docker run` or `apptainer run` will automatically launch `python ld_gpu.py` with any arguments passed to the container. Note that you will need to make sure to bind the input and output directories to directories within the container such that your container can read and write files to your host machine, as well as pass any relevant arguments to ensure that GPUs are made available to the processes. For example, your command might look like:

```
mpirun -n [num_processes] docker run --gpus all \
    -v ./input:/input \
    -v ./output/:/output/ \
    archer-ld [ARGS]
```

where `[ARGS]` represents the arguments described below.


Note that certain schedulers such as SLURM may implement other runners such as `srun` that support parallel jobs orchestrated by MPI. Please refer to the documentation for those schedulers to see how those commands work and how to implement them into your workflows, especially if you are running ARCHER-LD across multiple nodes.


### ARCHER-LD Example Commands

Some example commands can be found at the top of `ld_gpu.py`. We also provide some examples here:

#### Using the Conda environment (bare-metal)

Running Step 0 using your own environment:

```
mpirun -n 128 python ld_gpu.py \
    --step 0 \
    --input-vcf-folder ./input/all_vcfs/ \
    --convert-dir ./input/zarr_all_vcfs/ \
    --output-dir ./output/all_vcfs_LD/
```

Running Steps 1 and 2 using your own environment:

```
mpirun -n 9 python ld_gpu.py --step 1 2 \
    --gpu-memsize 12 \
    --gpu-overhead 1 \
    --phased \
    --ld-calc-threshes 0.7 \
    --input-zarr ./input/zarr_all_vcfs/combined_data.zarr.json \
    --output-dir ./output/all_vcfs_LD/
```


#### Using Docker

Running Step 0 using Docker:

```
mpirun -n 64 docker run --rm \
    -v ./input:/input \
    -v ./output:/output \
    archer-ld \
    --step 0 \
    --input-vcfs /input/chr1.vcf \
    --output-dir /output/chr1
```

Running Step 1 using Docker:

```
mpirun -n 9 docker run --rm --gpus all \
    -v ./input:/input \
    -v ./output:/output \
    archer-ld \
    --step 1 \
    --input-zarr /output/chr1/combined_data.zarr.json \
    --output-dir /output/chr1
```


#### Using Apptainer

Running Step 0 using Apptainer:

```
mpirun -n 64 apptainer run \
    -B ./input:/input \
    -B ./output:/output \
    archer-ld.sif \
    --step 0 \
    --input-vcfs /input/chr1.vcf \
    --output-dir /output/chr1
```

Running Step 1 using Apptainer

```
mpirun -n 9 apptainer run --nv \
    -B ./input:/input \
    -B ./output:/output \
    archer-ld.sif \
    --step 1 \
    --input-zarr /output/chr1/combined_data.zarr.json \
    --output-dir /output/chr1
```



### ARCHER-LD Arguments

All arguments can be found at the top of `ld_gpu.py`. We also provide a description of each here.

#### Step selection

The most critical argument is the `--step` argument, which determines which steps of ARCHER-LD to perform. Multiple steps can be performed, and the values passed are integer values of 0, 1, or 2, where:
- 0 = convert VCF or Plink files to Zarr (uses the CPU only, can use multiple processes)
- 1 = compute LD chunks using the converted Zarr store (primarily uses the GPU, needs CPU memory as well; designed to use multiple processes, one per GPU + one extra to send chunk information to each GPU)
- 2 = concatenate LD chunks generated by Step 1 (uses one process on the CPU only)

If this argument is not passed, it will be assumed that all steps should be performed (equivalent to passing `--step 0 1 2`).


#### Input/output arguments

Note that only one of the `--input-*` arguments can be passed at a time. The `--output-dir` argument is always mandatory, regardless of step.


| Argument                 	| Description                                                                                                                                                           	| Example usage(s)                                                                     	|
|--------------------------	|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------	|--------------------------------------------------------------------------------------	|
| `--input-vcfs`           	| Pass in multiple paths to VCF files.<br>Usually used in Step 0 to convert to Zarr.                                                                                    	| `--input-vcfs ./in_vcf/chr1.vcf ./in_vcf/chr2.vcf`                                   	|
| `--input-vcf-folder`     	| Pass in a path to a folder containing VCF files.<br>Usually used in Step 0 to convert to Zarr.                                                                        	| `--input-vcf-folder ./in_vcf/`                                                       	|
| `--input-plink-prefixes` 	| Pass in multiple Plink prefixes (omitting bed/bim/fam extension).<br>Usually used in Step 0 to convert to Zarr.                                                       	| `--input-plink-prefixes ./in_plink/chr1 ./in_vcf/chr2`                               	|
| `--input-plink-folder`   	| Pass in a path to a folder containing Plink files.<br>Usually used in Step 0 to convert to Zarr.                                                                      	| `--input-plink-folder ./in_plink/`                                                   	|
| `--input-zarr`           	| Pass in a path to a converted Zarr file. <br>Can be a virtual store (usually a `zarr.json` file).<br>Usually used in Step 1/2 to generate the LD matrix and metadata. 	| `--input-zarr ./converted/chr1.zarr`<br>`--input-zarr ./converted/chr_all.zarr.json` 	|
| `--output-dir`           	| Directory for final output of the chunks and LD matrix.                                                                                                               	| `--output-dir ./output_LD/`                                                          	|


#### Conversion arguments (only used for running Step 0)

| Argument                    	| Description                                                                                                               	| Default         	| Example usage(s)                    	|
|-----------------------------	|---------------------------------------------------------------------------------------------------------------------------	|-----------------	|-------------------------------------	|
| `--no-multivcf-parallel`    	| Disables parallel conversion of VCF files.<br>Only used if running Step 0 with VCF files.                                 	| [flag]          	| `--no-multivcf-parallel`            	|
| `--no-multiplink-parallel`  	| Disables parallel conversion of Plink files.<br>Only used if running Step 0 with Plink files.                             	| [flag]          	| `--no-multiplink-parallel`          	|
| `--convert-dir`             	| Directory to store the combined Zarr data.<br>If not passed, will default to `--output-dir`.                              	| N/A             	| `--convert-dir ./converted/`        	|
| `--convert-combined-prefix` 	| Prefix to use for the combined Zarr data.<br>If not passed, defaults to 'combined_data'.                                  	| 'combined_data' 	| `--convert-combined-prefix chr_all` 	|
| `--chunksize`               	| Chunk size for loading and processing data from the VCF, as<br>well as for the Zarr stores. Usually best as a power of 2. 	| 16384           	| `--chunksize 32768`                 	|


#### LD Computation arguments (only used for running Step 1)

| Argument                    	| Description                                                                                                                                                                                                                              	| Default         	| Example usage(s)                    	|
|-----------------------------	|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|-----------------	|-------------------------------------	|
| `--gpu-memsize`             	| The memory size of each GPU in GB.<br>(Can pass a smaller value, if desired).                                                                                                                                                            	| 12              	| `--gpu-memsize 16`                  	|
| `--gpu-overhead`            	| Amount of overhead to remove from the memory size per GPU.<br>Usually used to handle kernel and other overhead.<br>Rule of thumb is typically ~5% of the GPU memory, but you<br>can increase this value if ARCHER-LD runs out of memory. 	| 0.5             	| `--gpu-overhead 0.8`                	|
| `--dtype-used`              	| Data type to use for LD calculation.<br>Options are 'float16' and 'float32', but float16 support<br>is not guaranteed and likely leads to precision issues.                                                                              	| float32         	| `--dtype-used float32`              	|


#### Phasing Arguments (also only used if running Step 1)

Note that this argument set is a key determinant of the value you will receive - whether you receive the *phased* or *unphased* R^2 value as computed by ARCHER-LD. These arguments are mutually exclusive - setting one will determine the method used to determine how the data is treated. In a given analysis, **all** variants will be treated as phased or unphased.

These arguments play no role in Step 0 (data will be converted to Zarr as separate genotype arrays regardless of phasing with metadata that reports phasing information). These are all flags, so they have no additional values to pass beyond the argument itself.

| Argument              	| Description                                                                                                                                                                       	| Example usage(s)      	|
|-----------------------	|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|-----------------------	|
| `--phased`            	| Force ARCHER-LD to compute the phased R^2.<br>Strands will be treated as if they are phased across all variants.                                                                  	| `--phased`            	|
| `--unphased`          	| Force ARCHER-LD to compute the unphased R^2.<br>Strands will be added together to create a genotype matrix.                                                                       	| `--unphased`          	|
| `--determine-phasing` 	| Will use the metadata from Step 0 across ALL variants to determine <br>phasing. Time-consuming if the number of variants is large.                                                	| `--determine-phasing` 	|
| `--guess-phasing`     	| Will subsample 10000 sequential variants from a random starting point <br>from the Step 0 metadata and assess phasing for all of those variants. Usually fast<br>and the default behavior of ARCHER-LD. 	| `--guess-phasing`     	|

The default behavior if none of these are passed is equivalent to `--guess-phasing`.


#### Thresholding arguments

Note that these arguments are designed for extensibility in future versions of ARCHER-LD and can technically take multiple values. However, at present, only the R^2 LD value and a single threshold are supported and will be used. ARCHER-LD will currently fail if multiple values are passed to either of these arguments, and so we recommend only passing a single value to `--ld-calc-threshes` and ignoring `--ld-calc-methods` entirely.

| Argument             	| Description                                                                                                                                                                                                                                       	| Default 	| Example usage(s)         	|
|----------------------	|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|---------	|--------------------------	|
| `--ld-calc-threshes` 	| Threshold to use for ARCHER-LD during computation.<br>All values below this threshold will be zeroed out. <br>The current default is a value of 0.1.                                                                                              	| 0.1     	| `--ld-calc-threshes 0.8` 	|
| `--ld-calc-methods`  	| **NOTE: users should NOT currently use this argument.**<br><br>Methods/outputs to use for LD computation.<br>Currently only 'r2' (corresponding to R^2) is supported.<br>The default is the current intended behavior of ARCHER-LD. 	| r2      	| `--ld-calc-methods r2`   	|


#### Debug and override arguments

These arguments can be used to debug or override various values. For example, you can set the Zarr concurrency level or limit the number of blocks to compute in a run while debugging (or if you want to generate a cost estimate for ARCHER-LD, for example).


| Argument                     	| Description                                                                                                                                                                                                                                                                                     	| Default 	| Example usage(s)                   	|
|------------------------------	|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|---------	|------------------------------------	|
| `--zarr-async-concurrency`   	| Allows you to adjust the Zarr asynchronous concurrency level.<br>Passed directly to zarr.config.set({'async.concurrency': ...})<br>More details can be found [here](https://zarr.readthedocs.io/en/stable/user-guide/performance/#concurrent-io-operations).                                    	| 10      	| `--zarr-async-concurrency 128`     	|
| `--debug-max-blocks`         	| Automatically terminates ARCHER-LD during Step 1 once this number of blocks is reached.<br>Note that this does not change the actual total number of blocks needed for the LD matrix.<br>Useful if trying to generate a cost estimate (e.g., set to 16 to determine the cost by extrapolating). 	| None    	| `--debug-max-blocks 128`           	|
| `--debug-max-rows-per-block` 	| Maximum number of rows/variants to include in each chunk per GPU. <br>Can be useful if you want chunks to be a specific size lower than the GPU's <br>theoretical maximum (such as a round number).                                                                                             	| None    	| `--debug-max-rows-per-block 10000` 	|




## Additional Technical Details

For those technically inclined or those who want to know more, we provide some additional technical details below, but please let us know if you have any issues running ARCHER-LD or if you have any questions at all!


### Future Non-NVIDIA GPU support?

Currently, ARCHER-LD relies heavily on [CuPy](https://cupy.dev/) which uses CUDA Toolkit libraries to perform GPU operations. This inherently means that it supports and runs best on NVIDIA GPUs. We do not officially support other types of GPUs yet. 

However, [CuPy has experimental support for AMD GPUs through ROCm](https://docs.cupy.dev/en/stable/install.html#using-cupy-on-amd-gpu-experimental), though we have not performed any level of extensive testing on AMD GPUs. Users who have access to only AMD GPUs might consider this approach, though the reported limitations as of August of 2026 would imply that it would not work with ARCHER-LD.

We are also planning, in the future, to look into other options that would enable CUDA-based applications to run in a more universal manner, such as [chipStar](https://github.com/CHIP-SPV/chipStar). However, this is not currently a priority for us. Please let us know if such support would be critical to one of your use cases.


### Building the Docker image yourself

The configuration used to build the provided Docker image can be found in the `Dockerfile` file in this repository. We used the equivalent of the following command, and you can build your own version of the Docker image or build for a new platform (after cloning the repository) by running the same (removing or adding platforms as desired or even modifying the Dockerfile to fit your purposes). 

```
docker buildx build --platform=linux/amd64,linux/arm64 -t archer-ld .
```


### Containerization with MPI

Note that containerization with MPI can be more than a bit [challenging](https://apptainer.org/docs/user/latest/mpi.html). We have tested our container on a few different host systems and clusters, as well as both with Docker and with Apptainer, but there is a wide range of valid and even common host configurations, particularly for HPC systems. 

If you have issues with the Docker container, we first recommend, if at all possible, setting up a Conda environment and running bare-metal instead, but if this is not possible, then there are a few things that you could consider trying: 

1) We install OpenMPI within the container using the standard package repository (with `apt-get`) to try to mitigate build issues (as the arm64 build would often fail when trying to compile and install OpenMPI from source, likely due to emulation issues). The package repository has, in our testing as of August 2026, installed OpenMPI 4.1.6 (though this could change if the package repository is updated). If your host system runs a drastically different version of OpenMPI such as 5.x.x (or even a different MPI library altogether), this may fail to work or link processes properly (which may appear as all processes being reported as rank 0). You can try editing the Dockerfile and installing OpenMPI by compiling from source instead. 

2) Some host systems (and possibly our container's precompiled OpenMPI version) don't support the exact network infrastructure that OpenMPI (or your MPI library) might default to for communicating between processes. For OpenMPI, you can try changing MCA parameters by passing `--mca` to mpirun; we've found on some systems that bypassing UCX through something like `--mca btl vader,openib,self` can sometimes help with this issue, though you will likely have to experiment with this. Your sysadmin may also be able to help determine the best arguments to use.

3) If you are using Apptainer, it has certain features that can cause issues, such as [putting processes into unprivileged namespaces which are not shared](https://apptainer.org/docs/user/latest/mpi.html#using-sharens-mode). You can try adding the `--sharens` flag when running with Apptainer to see if this alleviates your issue.

Please also reach out to us if you are completely stuck and need help debugging.


### Schedulers

We have tested ARCHER-LD on HPC clusters running Slurm and LSF on very large datasets. If you are using a different scheduler, please feel free to reach out to us if you are having issues figuring out how to get ARCHER-LD to run on your cluster.


## ARCHER-LD Citation

The publication for ARCHER-LD is currently pending. We will post a citation here as soon as it is available. In the meantime, please reach out to us by email (see below) if you are planning to use ARCHER-LD for any analyses so that we can send you the citation as soon as it is available.


## Contact

Feel free to reach out if you are planning to use ARCHER-LD, as we would love to support your specific use case and find ways to make use of ARCHER-LD. 

Lead author:
- [Rachit Kumar, PhD](https://rachitk.com/): rachit [dot] kumar [at] pennmedicine [dot] upenn [dot] edu