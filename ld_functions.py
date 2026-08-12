# Description: Main LD metric computation functions

import cupy as cp
import numpy as np

from kernel_functions import *

# TODO: Convert these to C++ kernels (Kokkos?) for better performance
# via kernel fusions (though note that some of these ops are unfusable)


def compute_cross_rsquared_phased(chunk1, chunk2, kernels, dtype=cp.float32):
    """
    Computes the phased R^2 values for each variant in two (phased) chunks of data by
    computing D between the two chunks first, squaring it, and then dividing
    by the product of the major/minor allele frequencies for each pair
    For chunks of size m x n x 2 and p x n x 2, this will return an m x p matrix
    """
    # TODO: figure out how to do this without needing to double the storage
    # (some type of special kernel MUST exist for this, where we matmul and immediately reduce before allocating)
    # and also do the sample counts at the same time (at the vectorized level) so we ever only allocate n_var x n_var
    # For example, for the summed dot product, 
    # this would be something like cp.matmul.reduce(chunk1_cp, chunk2_cp, axis=0, out=ret)
    # but matmul.reduce is not implemented
    # basically write a kernel like https://siboehm.com/articles/22/CUDA-MMM

    # TODO: consider rewriting this in einsum for better performance
    # and to avoid needing to reallocate (maybe?)   

    # Send chunks to cupy
    # NOTE: we transpose both chunk1 and chunk2 before sending to cupy
    # for handling downstream
    chunk1 = chunk1.transpose(2,0,1)
    chunk2 = chunk2.transpose(2,1,0)
    chunk1_cp_0 = cp.asarray(chunk1[0], dtype=dtype) # m, n
    chunk2_cp_0 = cp.asarray(chunk2[0], dtype=dtype) # n, p
    chunk1_cp_1 = cp.asarray(chunk1[1], dtype=dtype) # m, n
    chunk2_cp_1 = cp.asarray(chunk2[1], dtype=dtype) # n, p

    # Compute dot product separately for each strand
    # (Requiring twice as much storage than the unphased, unfortunately)
    cp.nan_to_num(chunk1_cp_0, copy=False)
    cp.nan_to_num(chunk2_cp_0, copy=False)
    cp.nan_to_num(chunk1_cp_1, copy=False)
    cp.nan_to_num(chunk2_cp_1, copy=False)

    ret0 = cp.dot(chunk1_cp_0, chunk2_cp_0)
    ret1 = cp.dot(chunk1_cp_1, chunk2_cp_1)


    # Store the dot product sum in this out matrix by using only the first element
    # (hopefully this avoids needing to get more memory just for the sum)
    ret0 += ret1


    # Replace chunk1_cp_0 and chunk2_cp_0 with the original data with NaNs in it
    # TODO: replace this with a memcpy to the original variable to prevent
    # dereferencing and then rereferencing the memory (could use same address as before)
    del chunk1_cp_0, chunk2_cp_0


    # Reallocate chunk1_cp and chunk2_cp to store the dot product sum
    # NOTE: this implicitly assumes that any NaN on strand 1 is also on strand 2 
    # (there should never be a case where a genotype is partially missing)
    # We reallocate like this to avoid overhead when computing the inplace summation
    chunk1_cp_0 = cp.asarray(chunk1[0], dtype=dtype) # m, n
    chunk2_cp_0 = cp.asarray(chunk2[0], dtype=dtype) # n, p

    chunk1_cp_0 += chunk1_cp_1
    chunk2_cp_0 += chunk2_cp_1


    # Store genotype array in the second element of the chunk arrays
    # for reuse later
    cp.copyto(chunk1_cp_1, chunk1_cp_0)
    cp.copyto(chunk2_cp_1, chunk2_cp_0)

    # Compute the number of non-nan pairs for each variant
    cp.isnan(chunk1_cp_0, out=chunk1_cp_0)
    cp.isnan(chunk2_cp_0, out=chunk2_cp_0)

    cp.logical_not(chunk1_cp_0, out=chunk1_cp_0)
    cp.logical_not(chunk2_cp_0, out=chunk2_cp_0)


    # Compute the dot product of the two boolean arrays
    # and store into the ret1 matrix
    cp.dot(chunk1_cp_0, chunk2_cp_0, out=ret1)
    ret1 *= 2


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


    # Get the dot product of the two arrays to get the product of the allele frequencies
    # for each pair of variants, stored in ret1
    cp.dot(chunk1_cp_0[:,0][:, None], chunk2_cp_0[0,:][None,:], out=ret1)


    # Subtract ret1 from ret0 to get D for each pair of variants
    ret0 -= ret1
    


    ## Rsquared calculation
    # Square ret0 (D) and store result in ret1 (D^2)
    cp.square(ret0, out=ret1) # calculate D^2 inplace

    # Use kernel to compute R^2 from D^2
    kernels['rsq_from_dsq_kernel'](ret1, chunk1_cp_0[:,0], chunk2_cp_0[0,:], ret0.shape[1], ret1)

    # Clip values to 0, 1 to address numerical issues
    # and replace NaNs with 0 (as some values may be NaN due to division by 0 if the feature is constant)
    # and store in ret1
    cp.clip(ret1, 0, 1, out=ret1)
    cp.nan_to_num(ret1, copy=False)

    ## Rsq now in ret1
    cp.clip(ret1, 0, 1, out=ret1)


    ## Dmax and Dprime calculation
    # # NOTE: for now, this is commented out because it is extremely unstable to compute

    # # Compute Dmax from D, pA, and pB
    # # Store Dmax in ret1
    # kernels['dmax_kernel'](ret0, chunk1_cp_0[:,0], chunk2_cp_0[0,:], chunk2_cp_0.shape[1], ret1)

    # # Calculate D'
    # ret0 /= ret1

    # # Replace NaNs with 0 (as some values may be NaN due to division by nan if a strand has no valid alleles)
    # # (Also, some values may be inf or -inf due to division by 0)
    # # and then clip the values to 0, 1 to address numerical issues
    # cp.nan_to_num(ret0, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # cp.clip(ret0, -1, 1, out=ret0)

    # Delete chunk1_cp_0, chunk2_cp_0, chunk1_cp_1, chunk2_cp_1 to free up memory
    # TODO: if we aren't using ret0 for dmax, dprime, etc. we can probably delete this too
    # though we may want to return this to allow the user to calculate dprime later
    del chunk1_cp_0, chunk2_cp_0, chunk1_cp_1, chunk2_cp_1

    return ret1


def compute_cross_r2_unphased(chunk1, chunk2, kernels, impute=True, biased=False, dtype=cp.float32):
    """
    Computes the cross-correlation (squared) matrix between two chunks of data
    where each row represents a feature and each column represents a sample
    This means it will NOT compute the correlation for pairwise features within the same chunk
    For chunks of size m x n and p x n, this will return an m x p matrix
    """
    # Note that the chunks sent are the same as the phased version, of size
    # m x n x 2 and p x n x 2, but we only need to transpose the second chunk
    # but since this is unphased data, we operate only on the 0/1/2 vector, not the phased strands

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


    # Chunk2 genotype next
    chunk2_T = cp.asarray(chunk2_tp[0], dtype=dtype) # n, p
    chunk2_T_strand2 = cp.asarray(chunk2_tp[1], dtype=dtype) # n, p

    chunk2_T += chunk2_T_strand2
    del chunk2_T_strand2

    # TODO: figure out biased versus unbiased covariance
    if biased:
        dof = dtype(chunk1.shape[1])
    else:
        dof = dtype(chunk1.shape[1] - 1)

    # Center the data
    chunk1 -= cp.nanmean(chunk1, axis=1, dtype=dtype)[:, None]
    chunk2_T -= cp.nanmean(chunk2_T, axis=0, dtype=dtype)[None, :]

    # Impute NaNs with the mean of the non-NaN values (now centered to 0)
    # Note: imputing leads to artificially high LD for high-missingness data
    # TODO: figure out how to handle dropping NaNs if the user requests it
    if impute:
        cp.nan_to_num(chunk1, copy=False)
        cp.nan_to_num(chunk2_T, copy=False)
    else:
        raise NotImplementedError("Dropping of NaNs is not yet implemented for unphased data")

    # Get the sample covariance matrix
    ret = cp.empty((chunk1.shape[0], chunk2_T.shape[1]), dtype=dtype)
    cp.dot(chunk1, chunk2_T, out=ret)
    ret *= 1 / dof

    # Get the standard deviation of each feature in each of the chunks
    # Note that this is already centered, so we don't need to subtract the mean
    # as is done in the numpy std function
    # so we'll skip that computation and compute it ourselves
    cp.square(chunk1, out=chunk1)
    cp.square(chunk2_T, out=chunk2_T)

    ret /= cp.sqrt(chunk1.sum(axis=1) * (1 / dof))[:, None]
    ret /= cp.sqrt(chunk2_T.sum(axis=0) * (1 / dof))[None, :]

    # Square (for R2) and clip the values to 0, 1 to address numerical issues
    cp.square(ret, out=ret)
    cp.clip(ret, 0, 1, out=ret)

    # Replace NaNs with 0 (as some values may be NaN due to division by 0 if the feature is constant)
    cp.nan_to_num(ret, copy=False)

    # Delete chunk1 and chunk2_T to free up memory
    del chunk1, chunk2_T

    return ret