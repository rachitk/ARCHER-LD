# General imports
import os, glob
from functools import partial
import utils
import argparse

# Conversion and load imports
import convert_utils.vcf_convert_utils as v2z_utils
import convert_utils.plink_convert_utils as p2z_utils
import zarr

# Chunk calculation and merge imports
import cupy as cp
import cupyx as cpx
import numpy as np
from mpi4py import MPI
from mpi4py.util import pkl5
from utils import MPI_TAGS
from ld_functions import *
from kernel_functions import *
import scipy # only if saving in scipy sparse format



# TODO list:
# - Drop variants that are all NaN (likely padded variants at the end of a chromosome)
# - Remove scipy dependency by using CuPy sparse and storing numpy matrices of the things that define the sparse matrices
# - Improve file I/O (especially reading the Zarr data in chunks)
#   - Specifically, think of tricks like the ind1 == prev_ind1 trick for loading data
# - Add other LD metric computations (and implement kernels/functions for them)
# - Use kvikio for loading the Zarr data in chunks directly to the GPU (reduce transfer overhead)
# - Implement cuSPARSE-based kernel for covariance/correlation calculation (look at the math for this)
#   - Specifically, think of tricks for the kernels (maybe math along the lines of: https://stats.stackexchange.com/questions/120513/cross-correlation-for-very-sparse-binary-data)
# - Think about porting all of the underlying functions to C++ (OpenMP/Kokkos) for better performance and for more portability
# - Store chunk size in the output directory somewhere and also create some interface for accessing the store
#   - Need to store the indices for the variants as well (metadata for indices, positions, chromosomes, etc.)


parser = argparse.ArgumentParser(description='Compute LD matrices from phased/unphased VCF data')

# Example commands:

# Step 0 only, using a folder of VCFs
"""
mpiexec -n 128 python ld_gpu.py --step 0 \
    --input-vcf-folder ./input/1kg_highcoverage/all_vcfs_phased/ \
    --convert-dir ./input/zarr_1kg_data_allchr_md_phased \
    --output-dir ./output/1KG_blocks_allchr_md_phased
"""

# Step 1 and 2, using a Zarr made using a Step-0 only command (such as from above)
"""
mpiexec -n 7 python ld_gpu.py --step 1 2 \
    --input-zarr ./input/zarr_1kg_data_allchr_md_phased/combined_data.zarr.json \
    --output-dir ./output/1KG_blocks_allchr_md_phased_0.7 \
    --force-phased \
    --ld-calc-methods r2 \
    --ld-calc-threshes 0.7
"""


# Argument for which steps to perform
parser.add_argument('--step', type=int, default=None, nargs='+',
                    help='Which steps to perform (can pass multiple), where '
                    '0 = convert VCF to Zarr using the CPU, '
                    '1 = compute LD chunks using the GPU, '
                    '2 = concatenate LD chunks using the CPU. '
                    'Will run all steps together if this flag is not passed.')

# Mutually exclusive arguments for the input data
input_args_group = parser.add_mutually_exclusive_group(required=True)

input_args_group.add_argument('--input-vcfs', type=str, nargs='+',
                                help='Input VCF file(s) to convert to Zarr format (in order of desired concatenation)')
input_args_group.add_argument('--input-vcf-folder', type=str,
                                help='Input folder containing VCF files.')
input_args_group.add_argument('--input-plink-prefixes', type=str,
                                help='Input PLINK binary file(s) to load for LD computation. '
                                'Expected to be the prefix for each of the files to include (one prefix per bed/bim/fam trio).')
input_args_group.add_argument('--input-plink-folder', type=str,
                                help='Input folder containing PLINK binary files. '
                                'Expected to contain .bed, .bim, and .fam files for each dataset to include.')
input_args_group.add_argument('--input-zarr', type=str,
                                help='Input Zarr file to load for LD computation. Expected to be a Zarr array with the haplotype data. '
                                'Required if not converting VCFs to Zarr (i.e. step 0 is skipped by the user). '
                                'This can also be a kerchunk virtual store JSON file.')

# VCF-specific conversion arguments
parser.add_argument('--no-multivcf-parallel', action='store_true',
                    help='Do not attempt to convert multiple VCFs in parallel (if not passed, will do this)')

# Plink-specific conversion arguments
# parser.add_argument('--plink-convert-workers', type=int, default=0,
#                     help='Number of workers to use for each PLINK file conversion (default: 0, meaning will use only the main process on each rank). '
#                     'This can be used to speed up the conversion of PLINK files to Zarr format, but will increase memory usage.')
parser.add_argument('--no-multiplink-parallel', type=int, default=0,
                    help='Do not attempt to convert multiple Plink files in parallel (if not passed, will do this)')

# All conversion arguments (only used if step 0 is selected)
parser.add_argument('--convert-dir', type=str, default=None,
                    help='Directory to store the converted Zarr data. If not passed, defaults to --output-dir')
parser.add_argument('--convert-combined-prefix', type=str, default="combined_data",
                    help='Prefix for the combined Zarr file (default: combined_data)')
parser.add_argument('--chunksize', type=int, default=2**14,
                    help='Chunk size for loading and storing data from the VCFs (default: 2^14)')
parser.add_argument('--no-smart-sort', action='store_true',
                    help='Do not use smart sorting of the input files if a folder is passed (default: will do smart sorting). '
                    'This entails trying to find the best order in which to concatenate the input files based on the file names.')
                    
# LD computation arguments (only used if step 1 is selected)
parser.add_argument('--gpu-memsize', type=int, default=12,
                    help='Amount of memory to allocate for the GPU in GB (default: 12)')
parser.add_argument('--gpu-overhead', type=float, default=0.5,
                    help='Amount of overhead to remove from the memory size for the GPU in GB (default: 0.5)')
parser.add_argument('--dtype-used', type=str, default='float32',
                    help='Data type to use for the computation (default: float32). '
                    'float16 requires less memory usage but will lose precision. '
                    'Note: float32 is recommended for most use cases.')

step1_force_args = parser.add_mutually_exclusive_group()
step1_force_args.add_argument('--phased', action='store_true',
                    help='Compute the phased LD matrix (assume strands are phased properly across all variants)')
step1_force_args.add_argument('--unphased', action='store_true',
                    help='Compute the unphased LD matrix (no assumptions about phasing)')
step1_force_args.add_argument('--determine-phasing', action='store_true',
                    help='Force completely determining whether the data is phased or unphased based on Zarr metadata (check if all variants are phased; if so, treat the data as phased; otherwise, treat as unphased). Note that this can be very costly, so consider using --guess-phasing instead, which is much faster but less accurate.')
step1_force_args.add_argument('--guess-phasing', action='store_true',
                    help='Guess determination of whether the data is phased or unphased based on Zarr metadata (will take 10000 random sequential elements from the metadata and check if all variants are phased; if so, treat the data as phased; otherwise, treat as unphased). This is the default behavior.')


# LD computation threshold arguments
# TODO: implement other methods besides r2
parser.add_argument('--ld-calc-methods', type=str, nargs='+', default=['r2'],
                    help='Methods for LD computation (default: r2; for now, no other methods are implemented other than dprime, but that is unstable)')
parser.add_argument('--ld-calc-threshes', type=float, nargs='+', default=[0.1],
                    help='Thresholds for LD computation (default: 0.1) - must be of the same length as --ld-calc-methods')
                    

# Generic arguments (used by all or multiple steps)
parser.add_argument('--output-dir', type=str, required=True,
                    help='Output directory for the LD matrices and chunks (default: output)')
parser.add_argument('--zarr-async-concurrency', type=int, default=10,
                    help='Concurrency level for asynchronous Zarr operations (default: 10). '
                    'Normally Zarr only uses 10, but this is very conservative and can sometimes be higher on certain clusters and modern systems.')

# Debug arguments (used to do things like run only a few blocks for testing/cost computation)
parser.add_argument('--debug-max-blocks', type=int, default=None,
                    help='Maximum number of blocks to compute for debugging purposes (default: None, meaning no limit)')
parser.add_argument('--debug-max-rows-per-block', type=int, default=None,
                    help='Maximum number of rows to include in each block for debugging purposes (default: None, meaning no limit)')


args = parser.parse_args()




## MPI setup
comm = pkl5.Intracomm(MPI.COMM_WORLD)
rank = comm.Get_rank()
worldsize = comm.Get_size()
status = MPI.Status()

rankprint = partial(utils.rankprint, rank=rank)




## Set the shared arguments
output_dir = args.output_dir

# Data dtype used for computation and saving
if args.dtype_used == 'float32':
    dtype_used = np.float32
elif args.dtype_used == 'float16':
    dtype_used = np.float16
    rankprint("WARNING: Using float16 for computation can lead to loss of precision (and is not even supported on all GPUs), which can affect the accuracy of the LD calculations. Use with caution and consider using `float32`!", only_rank=0)
else:
    raise ValueError("Only float32 and float16 are supported as data types for LD computation or matrix saving!")

# Output block string (to be formatted with the block number)
out_file_formatstr = os.path.join(output_dir, "block{}.npz")


# Try to use zarrs for improved I/O performance, but if not available, just use zarr
# We will warn if the user doesn't have zarrs installed
try:
    import zarrs
    rankprint("Using `zarrs` to try to improve I/O performance...", only_rank=0)
    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
except ImportError:
    rankprint("`zarrs` not found, using base zarr for I/O (consider installing the `zarrs` package for improved performance)!", only_rank=0)



# Zarr asynchronous concurrency level 
# (unclear if the environment variable does anything downstream after already loaded)
# (but setting it anyways in case it helps - already see performance improvements with line 1)
zarr.config.set({"async.concurrency": args.zarr_async_concurrency})
os.environ["ZARR_ASYNC_CONCURRENCY"] = str(args.zarr_async_concurrency)


# Process steps passed in and the arguments that come along with them
if args.step is None:
    args.step = [0, 1, 2]

if 0 in args.step:
    convert_vcf = False
    convert_plink = False

    # Check if the input argument is a folder or a list of files
    if args.input_vcf_folder is not None:
        # Get list of VCFs in the folder and turn into a list
        input_vcfs = utils.get_vcf_files(args.input_vcf_folder,
                                         do_smart_sort=not args.no_smart_sort)
        
        if len(input_vcfs) == 0:
            raise ValueError(f"No VCF files found in folder {args.input_vcf_folder}! "
                             "Please provide a valid folder with VCF files to convert.")
        
        rankprint(f"Found {len(input_vcfs)} VCF files in folder {args.input_vcf_folder} to convert to Zarr format...", only_rank=0)
        convert_vcf = True

    elif args.input_vcfs is not None:
        # Convert the list of VCF files
        input_vcfs = args.input_vcfs
        convert_vcf = True


    elif args.input_plink_folder is not None:
        # Get list of PLINK prefixes in the folder and turn into a list
        input_plink_prefixes = utils.get_plink_prefixes(args.input_plink_folder)

        if len(input_plink_prefixes) == 0:
            raise ValueError(f"No PLINK files found in folder {args.input_plink_folder}! "
                             "Please provide a valid folder with PLINK binary files to load for LD computation.")
        
        rankprint(f"Found {len(input_plink_prefixes)} PLINK binary file sets in folder {args.input_plink_folder} to load for LD computation...", only_rank=0)
        convert_plink = True

    
    elif args.input_plink_prefixes is not None:
        # Convert the list of PLINK prefixes
        input_plink_prefixes = args.input_plink_prefixes
        convert_plink = True

    
    # No else needed because the above arguments are mutually exclusive but at least one is required


    # Conversion arguments
    multivcf_parallel = not args.no_multivcf_parallel
    multiplink_parallel = not args.no_multiplink_parallel
    plink_convert_workers = 0 #args.plink_convert_workers - currently doesn't work (doesn't play well with MPI, consider refactoring if needed)
    convert_combined_prefix = args.convert_combined_prefix
    chunksize = args.chunksize

    if args.convert_dir is not None:
        convert_folder = args.convert_dir
    else:
        rankprint("WARNING: No convert directory provided but step 0 passed, defaulting to output directory if needed...", only_rank=0)
        convert_folder = output_dir
    

# Check if the input Zarr file (or kerchunk virtual store, which is a JSON) is provided
# If so, skip the conversion step entirely and load the Zarr file
if args.input_zarr is not None:
    converted_filepath = args.input_zarr

    if 0 in args.step:
        rankprint("NOTE: Step 0 was passed, but skipping conversion step as input Zarr file is provided...", only_rank=0)
        args.step.remove(0)
else:
    if not (0 in args.step):
        raise ValueError("No input Zarr file provided and no conversion step (0) requested! "
                         "Please provide an input Zarr file or request the conversion step. "
                         "Note that requesting it again will not cause the data to be reconverted if done previously.")
    

if 1 in args.step:
    # LD computation arguments
    gpu_memsize = args.gpu_memsize
    gpu_overhead = args.gpu_overhead
    gpu_memsize_bytes = (gpu_memsize - gpu_overhead) * 1000**3

    # force phased and unphased
    force_phased = args.phased
    force_unphased = args.unphased
    guess_phasing = args.guess_phasing
    determine_phasing = args.determine_phasing

    
    # LD computation threshold arguments
    ld_calc_methods = args.ld_calc_methods
    ld_calc_threshes = args.ld_calc_threshes

    # Check if the number of thresholds is the same as the number of methods
    if len(ld_calc_methods) == len(ld_calc_threshes):
        ld_calc_threshes = {k: v for k, v in zip(ld_calc_methods, ld_calc_threshes)}
    else:
        raise ValueError("Number of LD computation methods and thresholds must be the same!")
        

if 2 in args.step:
    # Concatenation arguments
    # TODO: add some, for now none are needed
    pass




rankprint(f"Initialized process rank {rank}/{worldsize}")

# Synchronize ranks here to make sure all processes are ready
comm.Barrier()



### Actually perform the steps requested by the user below ###

if 0 in args.step:
    # Perform data conversion

    if convert_vcf:
        rankprint(f"Converting VCFs to Zarr genotype matrix...", only_rank=0)

        converted_filepath = v2z_utils.vcfs_to_zarr(input_vcfs, 
                                                    output_folder=convert_folder,
                                                    create_combined=True,
                                                    combined_prefix=convert_combined_prefix, 
                                                    overwrite_individual=False, 
                                                    overwrite_combined=False,
                                                    chunksize=chunksize, 
                                                    comm=comm, rank=rank, worldsize=worldsize,
                                                    parallel=multivcf_parallel)
        
    elif convert_plink:
        rankprint(f"Converting PLINK files to Zarr genotype matrix...", only_rank=0)

        converted_filepath = p2z_utils.plinks_to_zarr(input_plink_prefixes, 
                                                      output_folder=convert_folder,
                                                      create_combined=True,
                                                      combined_prefix=convert_combined_prefix, 
                                                      overwrite_individual=False, 
                                                      overwrite_combined=False,
                                                      chunksize=chunksize, 
                                                      comm=comm, rank=rank, worldsize=worldsize,
                                                      parallel=multiplink_parallel, 
                                                      n_workers=plink_convert_workers)
else:
    # Load the Zarr file
    rankprint("Skipping conversion step as input Zarr file is provided...", only_rank=0)

rankprint("Loading Zarr matrix...")


# Load the combined Zarr reference into memory on all ranks
# If JSON, load the (assumed) virtual store; otherwise, load as Zarr array
if converted_filepath.endswith('.json'):
    zarr_group = zarr.open('reference://', storage_options={'fo': converted_filepath}, mode='r')
else:
    zarr_group = zarr.open(converted_filepath, mode='r')


rankprint("Zarr matrix loaded as on-disk store (across ranks)!", only_rank=0)

# Variables for full genotype data and metadata
# Load the entire metadata into memory now since it's small 
# (and because there's a strange error where structured arrays are improperly supported, it seems)
fulldata = zarr_group['gt']
meta_fulldata = zarr_group['meta']


if(rank == 0):
    unfilt_meta_file = os.path.join(output_dir, "full_ldmat_unfiltered_variant_metadata.npy")

    if os.path.exists(unfilt_meta_file):
        rankprint(f"Skipping saving of unfiltered metadata as it already exists at {unfilt_meta_file}...")

    else:
        rankprint(f"Saving (unfiltered) metadata from Zarr matrix to disk at {unfilt_meta_file}...")

        # Make output directory (only if on rank 0 to avoid race conditions)
        os.makedirs(output_dir, exist_ok=True)

        # Save the unfiltered metadata to disk
        # The unfiltered metadata can allow you to load the data in chunks instead of all at once
        np.save(unfilt_meta_file, meta_fulldata)


# Compute some common values for the LD computation and downsteam processing
# Define values (to be acquired from the Zarr matrix)
arr_size = fulldata.shape

# Get data parameters
n_vars = arr_size[0]
n_samps = arr_size[1]


# Make sure all ranks have loaded the data before proceeding
comm.Barrier()

rankprint(f"Data is loaded and ready for LD computation! Detected {n_vars} variants (including padding) and {n_samps} samples.", only_rank=0)


# Check if cupy/cuda is available/accessible on this process
try:
    cp_avail = cp.cuda.is_available()
except Exception as e:
    cp_avail = False


# TODO: refactor the LD message-passing and such into another function (similar to how we did for conversion)

if not (1 in args.step):
    rankprint("Skipping LD computation per user request...", only_rank=0)
elif worldsize == 1:
    rankprint("Skipping LD computation as there is only one process (did you run this serially?). More than one needed...", only_rank=0)
elif not cp_avail:
    rankprint("Skipping LD computation as no CUDA-capable GPU was detected...", only_rank=0)
else:
    # Number of GPUs on this process
    # Use the one corresponding to the rank
    # (note we assume the number of GPUs is the same on all processes)
    n_gpus = cp.cuda.runtime.getDeviceCount()
    curr_gpu = rank % n_gpus

    # CUDA setup (only if not on rank 0, to allow one extra rank for sending data)
    if rank != 0:
        rankprint(f"Detected {n_gpus} GPUs on this node")
        cp.cuda.Device(curr_gpu).use()
        mempool = cp.get_default_memory_pool()

        rankprint(f"Initialized rank# {rank} (total {worldsize}) with GPU# {curr_gpu} (total {n_gpus}) (expected memsize used per GPU: {gpu_memsize_bytes} bytes)")
    else:
        rankprint(f"Rank 0 initialized, but does not use a GPU for computation or for sending information")


    # Create the elementwise kernels for thresholding
    # (this is done here to avoid recompiling the kernel on each iteration)
    avail_kernels = initialize_kernels(ld_calc_threshes, n_samps)


    # Make sure all ranks have initialized their GPUs before proceeding
    comm.Barrier()


    # Rank 0 will send the indices to the other ranks
    # and the other ranks will:
    # 1) receive the indices
    # 2) load the data
    # 3) compute the LD
    # 4) save the results to disk
    if rank == 0:

        # Check if the data is phased or unphased (if not passed by user)
        # Check if ANY variants are phased in the data; if so, treat the entire dataset as phased
        # (we can't handle mixed phased and unphased data in the same matrix for LD computation)
        # NOTE: heterozygotes that are unphased are coerced to 1/0 in the Zarr matrix, so unphased data
        # will assume that all dual heterozygotes contribute to the LD (overestimating it for unphased variants)
        if force_unphased:
            rankprint("Unphased LD will be computed...", only_rank=0)
            is_phased = False

        elif force_phased:
            rankprint("Phased LD will be computed...", only_rank=0)
            is_phased = True

        # Note: determining phasing is very costly, so we only do it on rank 0 if needed
        elif determine_phasing:
            rankprint("Determining phase of data to determine which type of LD... this can take some time, so consider passing --phased or --unphased to skip this step (or --guess-phasing if you're okay with a good approximation/guess)...", only_rank=0)
            phased_info = np.array(meta_fulldata[:]['N_PHASED'])
            is_phased = ((phased_info == n_samps) | (phased_info == -1)).all()
            del phased_info

        else:
            if guess_phasing:
                rankprint("Guessing phasing of data... this is a much faster alternative to --determine-phasing but less accurate", only_rank=0)
            else:
                rankprint("No phasing option passed, defaulting to guessing phasing (--guess-phasing) of data... this is a much faster alternative to --determine-phasing but less accurate", only_rank=0)
            if n_vars > 10000:
                random_index = np.random.randint(0, n_vars-10000) # to get a random sequential chunk of 10000 variants
            else:
                random_index = 0 # if there are less than 10000 variants, just take all of them
                rankprint(f"Warning: less than 10000 variants in the data, so using all variants to guess phasing (equivalent to --determine-phasing)", only_rank=0)
            phased_info = np.array(meta_fulldata[random_index:random_index+10000]['N_PHASED'])
            is_phased = ((phased_info == n_samps) | (phased_info == -1)).all()
            del phased_info


        rankprint(f"Data is being treated as {'' if is_phased else 'un'}phased data", only_rank=0)


        # Initialize the indices for the chunks
        n_rows_per_rank = utils.compute_chunksize(n_samps, gpu_memsize_bytes, dtype_used, is_phased=is_phased)

        if args.debug_max_rows_per_block is not None:
            n_rows_per_rank = min(n_rows_per_rank, args.debug_max_rows_per_block)
            rankprint(f"Debug max rows per block is set to {args.debug_max_rows_per_block}, using {n_rows_per_rank} rows per block for LD computation...", only_rank=0)
        else:
            rankprint(f"Using {n_rows_per_rank} rows per block for LD computation based on GPU memory size ({gpu_memsize_bytes/(1000**3)} GB) and data type ({np.dtype(dtype_used).itemsize} bytes)...", only_rank=0)

        # Send data continuously to all non-zero ranks until the end of the data
        # we will not use the GPU on rank 0 for now 
        # (or we will use it to process returned results from other ranks)
        # TODO: we will need to load the data in chunks from the file
        # TODO: send to ranks as they become free, rather than in round-robin, 
        # and send asynchronously (to overlap with computation)
        ind_gen = utils.upper_triangular_indices(n_vars, chunksize=n_rows_per_rank)
        expected_num_blocks = np.ceil(n_vars / n_rows_per_rank)
        expected_num_blocks = int(((expected_num_blocks ** 2) + expected_num_blocks) / 2)

        rankprint(f"Total expected blocks: {expected_num_blocks}", only_rank=0)

        for block_i, (ind1, ind2) in enumerate(ind_gen):
            # Define the chunks (rows) to send to the rank

            # TODO: do the loading and processing on the GPU if possible?
            if(os.path.exists(out_file_formatstr.format(block_i))):
                rankprint(f"Skipping existing file for block {block_i}")
                continue

            # Check if we have reached the debug max blocks limit and stop if so
            if args.debug_max_blocks is not None and block_i >= args.debug_max_blocks:
                rankprint(f"Reached debug max blocks limit of {args.debug_max_blocks}, stopping sending of data...", only_rank=0)
                break

            # Queue a receive for a free worker rank from all of the workers
            worker_data = comm.recv(source=MPI.ANY_SOURCE, status=status, tag=MPI_TAGS.REPORTBACK)
            worker_i = status.Get_source()

            # Send the data to the worker rank (including indices, whether to compute phased or unphased, and the number of rows per rank for loading)
            comm.send((block_i, ind1, ind2, is_phased, n_rows_per_rank), dest=worker_i, tag=MPI_TAGS.SENDDATA)
            rankprint(f"Sent chunk-pair {block_i} ({ind1}, {ind2}) ({(block_i+1)/expected_num_blocks:.3%}) to rank {worker_i} (total blocks: {expected_num_blocks})", only_rank=0)

        # Send None to all ranks to signal end of data
        # and allow them to exit the loop
        end_signal = (None, None, None, is_phased, n_rows_per_rank)
        for worker_i in range(1, worldsize):
            comm.send(end_signal, dest=worker_i, tag=MPI_TAGS.SENDDATA)


        # Make sure that rank 0 doesn't process data
        block_i, ind1, ind2 = None, None, None


    else:
        # Send a signal to the source rank that we are ready to receive data
        comm.send(None, dest=0, tag=MPI_TAGS.REPORTBACK)

        # Receive data on other ranks
        block_i, ind1, ind2, is_phased, n_rows_per_rank = comm.recv(source=0, status=status, tag=MPI_TAGS.SENDDATA)

        # Record last ind1 (so that we can avoid reloading data if it's the same as the previous one)
        prev_ind1 = -1
        chunk1 = None


        while block_i is not None:
            # Note: we load the data on each of the worker ranks
            # rankprint(f"Received chunk-pair {block_i} ({ind1}, {ind2}); loading chunks and computing {'phased' if is_phased else 'unphased'} LD")
            
            # TODO: different output file for each threshold or computation?
            # Note: checking if the file exists is done in the sender rank
            out_file = out_file_formatstr.format(block_i)

            # For each chunk, perform some very quick processing on the data if needed
            # NOTE: for missing, we do so for both strands in a chunk jointly
            # since if either are nan, we want both strands to be replaced
            # 1) replace all <0 with nan (are missing)
            # 2) replace all >1 with 1 (are multiallelic, will collapse all alt alleles into a single one)

            # Check if previous ind1 is the same as the current ind1
            # if so, don't need to reload that data
            # TODO: change this so that we load directly onto GPU 
            # (using zarr.config.enable_gpu() or Kvikio)
            # This would require a more comprehensive revamp of the computation of the LD matrix 
            # to avoid unnecessary transfers between host and device memory, but could be a good optimization to reduce the overall runtime

            if(ind1 != prev_ind1):
                chunk1 = np.asarray(fulldata[ind1:ind1+n_rows_per_rank], dtype=dtype_used)
                chunk1.clip(max=1, out=chunk1) # equivalent to chunk1[chunk1 > 1] = 1 without copy for masking
                chunk1[chunk1 < 0] = np.nan # TODO: figure out how to do this inplace (Numba kernel?)

                # Update previous index
                prev_ind1 = ind1

            # Always load chunk2 
            # (since we go row-wise in sending data, ind2 will essentially always be different)
            chunk2 = np.asarray(fulldata[ind2:ind2+n_rows_per_rank], dtype=dtype_used)
            chunk2.clip(max=1, out=chunk2) # equivalent to chunk2[chunk2 > 1] = 1 without copy for masking
            chunk2[chunk2 < 0] = np.nan # TODO: figure out how to do this inplace (Numba kernel?)

            rankprint(f"Loaded chunk-pair {block_i} ({ind1}, {ind2}) from disk - computing {'phased' if is_phased else 'unphased'} LD now!")

            # Compute the cross-correlation matrix 
            # with some optimizations based on assumptions
            # TODO: break up R^2 calculation if phased into D, D', etc.

            # If the data is unphased (no variants are phased), 
            # we approximate LD with the square of the Pearson correlation
            # of the 0/1/2 genotype array (summed the phased over the last axis)
            if(not is_phased):
                r2 = compute_cross_r2_unphased(chunk1, chunk2, avail_kernels, dtype=dtype_used)

            # If the data is phased, we compute R^2 by using the R^2 formula
            # which is based on the allele frequencies and dot product of strands
            else:
                r2 = compute_cross_rsquared_phased(chunk1, chunk2, avail_kernels, dtype=dtype_used)
                
            # Delete chunk2 to try to free memory for the sparse LD matrix 
            # (chunk1 may be reused if the next ind1 is the same)
            del chunk2

            rankprint(f"Computed chunk-pair {block_i} ({ind1}, {ind2}) LD matrix - saving to disk now!")

            if('threshold_r2_kernel' in avail_kernels.keys()):
                # Threshold the R2 matrix in place
                # Use a custom elementwise kernel using CuPy (or C++) for
                # computing inplace thresholding (to avoid memory allocation)
                avail_kernels['threshold_r2_kernel'](r2, r2)

                # Sparse save
                # (using scipy.sparse.save_npz after converting, then copying memory to host)
                # Note, we are saving the matrix in CSR format
                # but we start by getting the CSC of the tranpose as this avoids a copy
                # to make the array Fortran-contiguous (TODO: PR this optimization to CuPy?)
                # This has some calculation overheads but it's basically free and speeds up the overall program
                r2_sparse = cpx.scipy.sparse.csc_matrix(r2.T).T

                # Save the sparse matrix to disk using scipy (after copying to host memory)
                # TODO: save directly to disk from GPU to avoid transfer
                # (using zarr.config.enable_gpu() or Kvikio)
                scipy.sparse.save_npz(out_file, r2_sparse.get())
                
                rankprint(f"Saved chunk-pair {block_i} ({ind1}, {ind2}) r2 matrix to {out_file}!")

                # Delete sparse matrix r2_sparse to free up memory
                del r2_sparse

            
            # Dereference to indicate that memory can be reused
            # (do not call mempool.free_all_blocks() here)
            # (Need to delete r2_sparse or r2 will be indirectly referenced and not freed)
            del r2

            # Send a signal to the source rank that we are ready to receive data
            comm.send(None, dest=0, tag=MPI_TAGS.REPORTBACK)

            # Receive data again until we get None
            block_i, ind1, ind2, is_phased, n_rows_per_rank = comm.recv(source=0, status=status, tag=MPI_TAGS.SENDDATA)
        
        # Once we're done, clean up memory so that rank 0 has more (if needed) by deleting chunk1
        del chunk1



# Make sure LD computation is completed by every rank before continuing
comm.Barrier()


# After completing the LD matrix calculation, Rank 0 will:
# 1) Save the filtered metadata to disk
# 2) load all of the results
# 3) concatenate them into a single matrix
# 4) filter the matrix data based on the metadata
# 5) save the final matrix and metadata to disk
if(rank == 0):
    rankprint("Rank 0 is loading some data for filtered metadata saving (always done) and sparse block concatenation (only if step 2 enabled)...")

    meta_fulldata = meta_fulldata[:] # load into memory since needed to compute and filter

    # Load the position attribute from the Zarr matrix
    # position == -1 for all padded variants
    # Save the metadata to disk for future reference
    var_pos_data = meta_fulldata['POS']
    true_var_mask = var_pos_data != -1
    n_real_vars = true_var_mask.sum()

    # Save the variant metadata after filtering to disk as a set of NPZ files
    # (this will allow us to load the metadata in the future)
    filt_meta_file = os.path.join(output_dir, "full_ldmat_filtered_variant_metadata.npy")

    if os.path.exists(filt_meta_file):
        rankprint(f"Skipping saving of filtered metadata as it already exists at {filt_meta_file}...")
    else:
        rankprint(f"Saving filtered variant metadata for {n_real_vars} variants (after filtering) to disk at {filt_meta_file}...")

        # below only works for Zarr array, but we've loaded into memory as Numpy array already (since Zarr 3.0 doesn't support structured arrays properly)
        # filt_meta = meta_fulldata.get_mask_selection(true_var_mask)

        # Method for in-memory Numpy array is just to mask directly
        filt_meta = meta_fulldata[true_var_mask]
        np.save(filt_meta_file, filt_meta)
        rankprint(f"Saved filtered variant metadata for {n_real_vars} variants (after filtering) to disk at {filt_meta_file}!")

    # Perform some final cleanup and processing of the matrix on rank 0
    # including getting metadata from the Zarr matrix and using it to filter the data
    # as well as concatenating chunks of scipy sparse matrix into a single file
    out_scipy_file = os.path.join(output_dir, "full_ldmat_filtered_csr.npz")


    if not (2 in args.step):
        rankprint("Skipping filtering and concatenation of the chunks into a single file at user request...")
    elif os.path.exists(out_scipy_file):
        rankprint(f"Skipping concatenation as the final matrix {out_scipy_file} already exists...")
    else:
        rankprint("Concatenating the final matrix chunks (after filtering away padded elements) into a single file (slow to save to disk)...")


        # Collect all block files and load them
        # (this is done on rank 0 to avoid memory issues)

        # Count number of files meeting the output file formatstring
        # (we count rather than just using the string to indirectly check for missing files)
        num_blocks = len(glob.glob(out_file_formatstr.format('*')))
        rankprint(f"Loading sparse block files... {num_blocks} total blocks detected")

        if num_blocks == 0:
            raise ValueError("No block files detected! Did you pass the correct --output-dir and have you run step 1 previously? "
                             "Were there GPUs available when you ran step 1?")

        scipy_list = []
        for i in range(num_blocks):
            print(f"[0]: Loading block {i}...", end='\r')
            scipy_list.append(scipy.sparse.load_npz(out_file_formatstr.format(i)))

        rankprint("Finished loading all sparse block files.")
        
        total_nnz = sum([x.nnz for x in scipy_list])

        # Initialize a scipy sparse matrix to hold the final data
        # (we will concatenate the blocks into this matrix)
        # Note: we will use CSR format for the final matrix
        # (as this is the most efficient for row-wise access)
        scipy_out_rows = np.zeros(total_nnz, dtype=np.uint32)
        scipy_out_cols = np.zeros(total_nnz, dtype=np.uint32)
        scipy_out_data = np.zeros(total_nnz, dtype=dtype_used)
        


        # Iterate over the blocks and filter the data based on the metadata, then assign to a new Zarr matrix
        # (we skip the lower triangle and the diagonal here, recognizing that the data is symmetric)
        # Here, curr_nrows and curr_ncols are the current row and column indices of the UNFILTERED matrix
        curr_nrows = 0
        curr_ncols = 0
        curr_block_i = 0
        curr_rowblock_i = 0
        curr_colblock_i = 0

        curr_nrows_real = 0
        curr_ncols_real = 0

        curr_nnz = 0

        tot_col_offsets_per_col_i = {}
        real_col_offsets_per_col_i = {}
        tot_row_offsets_per_row_i = {}
        real_row_offsets_per_row_i = {}

        rankprint("Filtering and assigning data to a final single matrix...")

        # TODO: compute all the offsets for each block and masks thereof in advance
        # and then multiprocess to assign the data to the final matrix (since Zarr supports
        # multiple-process-writing). This will speed up the process of writing the final matrix
        # as the slowest part is writing to chunks of a Zarr array

        # TODO: also add metadata to the Zarr array of which blocks have been processed to allow skipping later
        # (will allow us to skip loading the scipy sparse matrices themselves too)

        # TODO: consider moving basically all of this metadata-filtering to the original passing
        # of the data to the ranks (and then save from the ranks directly into the Zarr output array)
        # this prevents needing to reload the sparse matrix and allows parallelization of the storage process

        # TODO: just make this output a scipy sparse array and index into it directly to assign elements
        # instead of using the Zarr array (to try to reduce the disk overhead)

        while curr_nrows < n_vars:
            curr_block_nrows = scipy_list[curr_block_i].shape[0]

            # Get which variants in this row set are real
            row_var_real = true_var_mask[curr_nrows:curr_nrows+curr_block_nrows]
            nreal_row_vars = row_var_real.sum()

            while curr_ncols < n_vars:
                if curr_ncols < curr_nrows:
                    # Skip the lower triangle
                    tot_colblock_offset = tot_col_offsets_per_col_i[curr_colblock_i]
                    real_colblock_offset = real_col_offsets_per_col_i[curr_colblock_i]

                    curr_ncols += tot_colblock_offset
                    curr_ncols_real += real_colblock_offset

                    curr_colblock_i += 1
                    continue

                # Load the next block 
                # (TODO: load the actual file, rather than just indexing into the list)
                curr_block = scipy_list[curr_block_i]
                curr_block_ncols = curr_block.shape[1]

                # Get which variants in this column set are real
                col_var_real = true_var_mask[curr_ncols:curr_ncols+curr_block_ncols]
                nreal_col_vars = col_var_real.sum()

                # Use the real variant indices to filter the data
                curr_block = curr_block[row_var_real][:, col_var_real]

                # If this block is a diagonal one, 
                # remove the lower triangle and diagonal from it
                # (note, converts to COO format if used, consider coercing CSR if we use scipy sparse output instead of Zarr)
                if curr_ncols == curr_nrows:
                    curr_block = scipy.sparse.triu(curr_block, k=1)

                curr_block = curr_block.tocoo()

                rankprint(f"Processing block {curr_block_i} with real shape ({nreal_row_vars}, {nreal_col_vars}) into final matrix at upper-left position ({curr_nrows_real}, {curr_ncols_real})")

                # Assign the block to the final matrix using coordinate indexing
                scipy_out_rows[curr_nnz:curr_nnz+curr_block.nnz] = curr_block.row + curr_nrows_real
                scipy_out_cols[curr_nnz:curr_nnz+curr_block.nnz] = curr_block.col + curr_ncols_real
                scipy_out_data[curr_nnz:curr_nnz+curr_block.nnz] = curr_block.data

                # curr_block = curr_block.tocoo()
                # zarr_out.set_coordinate_selection((curr_block.row + curr_nrows_real, 
                #                                    curr_block.col + curr_ncols_real),
                #                                    curr_block.data)


                # Update ncols offset and block index
                curr_ncols += curr_block_ncols
                curr_ncols_real += nreal_col_vars
                tot_col_offsets_per_col_i[curr_colblock_i] = curr_block_ncols
                real_col_offsets_per_col_i[curr_colblock_i] = nreal_col_vars
                curr_block_i += 1
                curr_nnz += curr_block.nnz
                curr_colblock_i += 1

            
            # Update nrows offset and reset ncols offset
            curr_nrows += curr_block_nrows
            curr_nrows_real += nreal_row_vars
            curr_rowblock_i += 1
            tot_row_offsets_per_row_i[curr_rowblock_i] = curr_block_nrows
            real_row_offsets_per_row_i[curr_rowblock_i] = nreal_row_vars
            curr_ncols = 0
            curr_ncols_real = 0
            curr_colblock_i = 0

        scipy_out = scipy.sparse.coo_matrix((scipy_out_data, (scipy_out_rows, scipy_out_cols)), shape=(n_real_vars, n_real_vars))

        # Check to make sure the final matrix has the correct number of non-zero elements
        assert scipy_out.nnz == total_nnz, f"Final matrix has {scipy_out.nnz} non-zero elements, expected {total_nnz}! Please report this bug to the developers."

        # Save the final matrix to disk
        rankprint("Converting final matrix to CSR and saving to disk... Note that this can take a long time.")
        scipy_out_csr = scipy_out.tocsr()
        scipy.sparse.save_npz(os.path.join(output_dir, "full_ldmat_filtered_csr.npz"), scipy_out_csr)
        # scipy.sparse.save_npz(os.path.join(output_dir, "full_ldmat_filtered_coo.npz"), scipy_out)

        # The metadata can be loaded with
        # npzfile = np.load(os.path.join(output_dir, "full_ldmat_filtered_variant_metadata.npz"))
        # Generally, this is essentially instantaneous.

        # The LD matrix can be loaded with
        # ldmat = scipy.sparse.load_npz(os.path.join(output_dir, "full_ldmat_filtered_csr.npz"))
        # Generally, this takes no more than a few minutes.

else:
    rankprint("Non-Rank 0 process(es) terminating as only Rank 0 performs step 2 (if requested)... Note that clusters will still charge you for these resources, so consider separating Step 1 and 2!", only_rank=1)

rankprint(f"All steps have completed (or have been skipped)...", only_rank=0)