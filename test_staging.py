# Description: A file that is used as a staging/debugging ground for
# code snippets that are being developed for the project, especially
# those that use the CPU only (as it is a bit easier to debug here)


import numpy as np
import cupy as cp

import ipdb

from kernel_functions import create_dmax_kernel, create_rsq_from_dsq_kernel
from utils import compute_chunksize

dtype = cp.float32

cp.cuda.Device(0).use()
mempool = cp.get_default_memory_pool()


n_samps = 57170
gpu_memmax_test = 32 * 1000**3
limit_gpu_mem = 180 * 1000**3
do_phased = False

chunksize = compute_chunksize(n_samps=n_samps, gpu_memsize_bytes=gpu_memmax_test, dtype_used=dtype, is_phased=do_phased)

mempool = cp.get_default_memory_pool()

if limit_gpu_mem is not None:
    with cp.cuda.Device(0):
        mempool.set_limit(size=limit_gpu_mem)


print(f"Chunk Size: {chunksize} x {n_samps}, gpu mem max: {gpu_memmax_test / 1000**3} GB, dtype size: {dtype().nbytes}, phased: {do_phased}, limit_gpu_mem: {limit_gpu_mem / 1000**3} GB")


## TODO: figure out why increasing the number of 1s increases the D' distribution 
# (and also why D' is >> 1 for some pairs)

# Set random seed
np.random.seed(0)

# Test the covariance/correlation calculation
arr_size = [chunksize, n_samps, 2]
test_arr = np.zeros(arr_size, dtype=np.float32)
# test_arr = np.random.rand(*arr_size) < 0.1
# test_arr = test_arr.astype(np.float32)

chunk1 = np.flip(test_arr, axis=0)
chunk2 = chunk1

# nan_mask1 = np.random.rand(*chunk1.shape[0:-1]) < 0.01
# chunk1[nan_mask1] = np.nan


# Cupy events for timing
start_gpu = cp.cuda.Event()
end_gpu = cp.cuda.Event()

start_gpu.record()

def print_mem(do_free=False):
    print(f"Used memory: {mempool.used_bytes() / 1000**3} GB, total memory: {mempool.total_bytes() / 1000**3} GB")
    if do_free:
        mempool.free_all_blocks()


ipdb.set_trace()


if(not do_phased):
    biased = False
    impute = True

    print("Start")
    print_mem()

    # # Start by summing the two strands to get the unphased data
    # # This is done on the CPU to avoid memory overhead 
    # # (TODO: consider moving back to GPU?)
    # # now m x n and p x n
    # chunk1 = chunk1.sum(axis=-1)
    # chunk2 = chunk2.sum(axis=-1)

    # print("Initial summation")
    # print(mempool.used_bytes())
    # print(mempool.total_bytes())

    # # Send to device and cast to a float32 dtype here for the computations downstream
    # # Note that we transpose chunk2 here BEFORE sending it to the GPU
    # chunk1 = cp.asarray(chunk1, dtype=cp.float32)
    # chunk2_T = cp.asarray(chunk2.T, dtype=cp.float32)

    # Allocate and sum
    # Send chunks to cupy
    # NOTE: we transpose both chunk1 and chunk2 before sending to cupy
    # for handling downstream
    chunk1_tp = chunk1.transpose(2,0,1)
    chunk2_tp = chunk2.transpose(2,1,0)

    # Get genotypes for each chunk separately to only need to load three strands max at a time
    # Chunk1 genotype first
    chunk1 = cp.asarray(chunk1_tp[0], dtype=dtype) # m, n
    chunk1_strand2 = cp.asarray(chunk1_tp[1], dtype=dtype) # m, n

    chunk1 += chunk1_strand2
    del chunk1_strand2

    print("Chunk1 genotype made")
    print_mem()

    # Chunk2 genotype next
    chunk2_T = cp.asarray(chunk2_tp[0], dtype=dtype) # n, p
    chunk2_T_strand2 = cp.asarray(chunk2_tp[1], dtype=dtype) # n, p

    chunk2_T += chunk2_T_strand2

    print("Chunk2 genotype made")
    print_mem()

    del chunk2_T_strand2

    print("Chunk2 strand deleted ")
    print_mem()

    # TODO: figure out biased versus unbiased covariance
    if biased:
        dof = dtype(chunk1.shape[1])
    else:
        dof = dtype(chunk1.shape[1] - 1)

    # Center the data
    chunk1 -= cp.nanmean(chunk1, axis=1, dtype=dtype)[:, None]
    chunk2_T -= cp.nanmean(chunk2_T, axis=0, dtype=dtype)[None, :]

    print("Center data")
    print_mem()

    # Impute NaNs with the mean of the non-NaN values (now centered to 0)
    # Note: imputing leads to artificially high LD for high-missingness data
    # TODO: figure out how to handle dropping NaNs if the user requests it
    if impute:
        cp.nan_to_num(chunk1, copy=False)
        cp.nan_to_num(chunk2_T, copy=False)
    else:
        raise NotImplementedError("Dropping of NaNs is not yet implemented for unphased data")
    
    print("Impute NaNs")
    print_mem()

    # Get the sample covariance matrix
    ret = cp.empty((chunk1.shape[0], chunk2_T.shape[1]), dtype=dtype)

    print("Allocate ret")
    print_mem()

    cp.dot(chunk1, chunk2_T, out=ret)

    print("Dot product")
    print_mem()

    ret *= 1 / dof

    print("Get cov matrix")
    print_mem()

    # Get the standard deviation of each feature in each of the chunks
    # Note that this is already centered, so we don't need to subtract the mean
    # as is done in the numpy std function
    # so we'll skip that computation and compute it ourselves
    cp.square(chunk1, out=chunk1)
    cp.square(chunk2_T, out=chunk2_T)

    print("Square data inplace")
    print_mem()

    ret /= cp.sqrt(chunk1.sum(axis=1) * (1 / dof))[:, None]

    print("Scale ret by chunk1 std")
    print_mem()

    ret /= cp.sqrt(chunk2_T.sum(axis=0) * (1 / dof))[None, :]

    print("Scale ret by chunk2 std")
    print_mem()

    # Square (for R2) and clip the values to 0, 1 to address numerical issues
    cp.square(ret, out=ret)
    cp.clip(ret, 0, 1, out=ret)

    print("Square ret and clip")
    print_mem()

    # Replace NaNs with 0 (as some values may be NaN due to division by 0 if the feature is constant)
    cp.nan_to_num(ret, copy=False)

    print("Finish up")
    print_mem()

    # Delete chunk1 and chunk2_T to free up memory
    del chunk1, chunk2_T

    end_gpu.record()
    end_gpu.synchronize()
    gpu_time = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    print(f"GPU time: {gpu_time / 1000} seconds")

    ipdb.set_trace()



else:
    print("Start")
    print_mem()

    dmax_kernel = create_dmax_kernel()
    rsq_from_dsq_kernel = create_rsq_from_dsq_kernel()

    # Send chunks to cupy
    # NOTE: we transpose both chunk1 and chunk2 before sending to cupy
    # for handling downstream
    chunk1 = chunk1.transpose(2,0,1)
    chunk2 = chunk2.transpose(2,1,0)
    chunk1_cp_0 = cp.asarray(chunk1[0], dtype=dtype) # m, n
    chunk2_cp_0 = cp.asarray(chunk2[0], dtype=dtype) # n, p
    chunk1_cp_1 = cp.asarray(chunk1[1], dtype=dtype) # m, n
    chunk2_cp_1 = cp.asarray(chunk2[1], dtype=dtype) # n, p


    print("Initial allocation")
    print_mem()

    # Compute dot product separately for each strand
    # (Requiring twice as much storage than the unphased, unfortunately)
    cp.nan_to_num(chunk1_cp_0, copy=False)
    cp.nan_to_num(chunk2_cp_0, copy=False)
    cp.nan_to_num(chunk1_cp_1, copy=False)
    cp.nan_to_num(chunk2_cp_1, copy=False)

    print("nantonum")
    print_mem()

    ret0 = cp.dot(chunk1_cp_0, chunk2_cp_0)
    ret1 = cp.dot(chunk1_cp_1, chunk2_cp_1)

    print("Matmul over two strands")
    print_mem()

    # Store the dot product sum in this out matrix by using only the first element
    # (hopefully this avoids needing to get more memory just for the sum)
    ret0 += ret1

    print("Sum of ret0 and ret1 and store in ret0")
    print_mem()

    # Delete unnecessary variables now to free up some memory
    # TODO: replace this with a memcpy to the original variable to prevent
    # dereferencing and then rereferencing the memory (could use same address as before)
    del chunk1_cp_0, chunk2_cp_0

    print("Delete chunks")
    print_mem()

    # Reallocate chunk1_cp and chunk2_cp to store the dot product sum
    # NOTE: this implicitly assumes that any NaN on strand 1 is also on strand 2 
    # (there should never be a case where a genotype is partially missing)
    # We reallocate like this to avoid overhead when computing the inplace summation
    chunk1_cp_0 = cp.asarray(chunk1[0], dtype=dtype) # m, n
    chunk2_cp_0 = cp.asarray(chunk2[0], dtype=dtype) # n, p

    print("Realloc")
    print_mem()

    chunk1_cp_0 += chunk1_cp_1
    chunk2_cp_0 += chunk2_cp_1

    print("Sum")
    print_mem()

    # Store genotype array in the second element of the chunk arrays
    # for reuse later
    cp.copyto(chunk1_cp_1, chunk1_cp_0)
    cp.copyto(chunk2_cp_1, chunk2_cp_0)

    # Compute the number of non-nan pairs for each variant
    cp.isnan(chunk1_cp_0, out=chunk1_cp_0)
    cp.isnan(chunk2_cp_0, out=chunk2_cp_0)

    cp.logical_not(chunk1_cp_0, out=chunk1_cp_0)
    cp.logical_not(chunk2_cp_0, out=chunk2_cp_0)

    print("Sum and (not)isnan")
    print_mem()

    # Compute the dot product of the two boolean arrays
    # and store into the ret1 matrix
    cp.dot(chunk1_cp_0, chunk2_cp_0, out=ret1)
    ret1 *= 2

    print("Dot for num non-nan pairs")
    print_mem()

    # Get pAB for each pair by dividing the dot product by the number of samples
    # (this is the number of pairs that are not missing in either variant)
    ret0 /= ret1


    # Get the allele frequencies for each variant
    # Noting that the genotype array is stored in chunkN_cp_1 now
    chunk1_cp_1 /= 2
    chunk2_cp_1 /= 2

    # Compute pA and pB and store within chunk1_cp_0 and chunk2_cp_0
    # (storing in the first columns of each, noting chunk2_cp is transposed)
    cp.nanmean(chunk1_cp_1, axis=1, out=chunk1_cp_0[:,0])
    cp.nanmean(chunk2_cp_1, axis=0, out=chunk2_cp_0[0,:])

    print("Compute allele frequencies")
    print_mem()

    # Get the dot product of the two arrays to get the product of the allele frequencies
    # for each pair of variants, stored in ret1
    cp.dot(chunk1_cp_0[:,0][:, None], chunk2_cp_0[0,:][None,:], out=ret1)

    print("Get product of allele frequencies and store in ret1")
    print_mem()


    # Subtract ret1 from ret0 to get D for each pair of variants
    ret0 -= ret1

    print("Subtract and then copy off ret0")
    print_mem()


    # Compute Rsq from D
    cp.square(ret0, out=ret1) # calculate D^2 inplace

    print("Square D and store in ret1")
    print_mem()

    # Use kernel to compute R^2 from D^2
    rsq_from_dsq_kernel(ret1, chunk1_cp_0[:,0], chunk2_cp_0[0,:], ret0.shape[1], ret1)

    print("Rsq in ret1")
    print_mem()


    cp.clip(ret1, 0, 1, out=ret1)
    cp.nan_to_num(ret1, copy=False)

    print("Clip ret1 back into ret1")
    print_mem()

    ## Rsq now in ret1



    # Compute Dmax from D, pA, and pB
    # Store Dmax in ret1
    dmax_kernel(ret0, chunk1_cp_0[:,0], chunk2_cp_0[0,:], chunk2_cp_0.shape[1], ret1)

    print("Use dmax kernel")
    print_mem()


    # Calculate D'
    ret0 /= ret1


    print("Calculate D prime")
    print_mem()


    end_gpu.record()
    end_gpu.synchronize()
    gpu_time = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    print(f"GPU time: {gpu_time / 1000} seconds")

    ipdb.set_trace()