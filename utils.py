# Description: Utility functions for the project

import numpy as np
import os, glob, re

# Generator for indices of the upper triangular part of a matrix
# TODO maybe make this a class to implement length
def upper_triangular_indices(n, chunksize=1):
    chunksize = int(chunksize)
    chunksize = min(n, chunksize)
    
    for i in range(0, n, chunksize):
        for j in range(0, n, chunksize):
            if j < i:
                continue
            
            yield i, j


# Convenience function for printing with rank
def rankprint(string, rank=0, only_rank=None):
    if only_rank is None or rank == only_rank:
        print(f"[{rank}]: {string}", flush=True)


# Function to get a list of VCF files in a directory
def get_vcf_files(directory, do_smart_sort=True):
    """
    Get a list of VCF files in a directory
    (will also search for vcf.gz files)
    """
    vcf_files = glob.glob(os.path.join(directory, "*.vcf"))
    vcf_files.extend(glob.glob(os.path.join(directory, "*.vcf.gz")))

    vcf_files = [os.path.basename(vcf_file) for vcf_file in vcf_files]

    if do_smart_sort:
        vcf_files = smart_sort_filenames(vcf_files)

    vcf_files = [os.path.join(directory, vcf_file) for vcf_file in vcf_files]

    return vcf_files

# Function to get a list of PLINK prefixes in a directory
def get_plink_prefixes(directory, do_smart_sort=True):
    """
    Get a list of PLINK prefixes in a directory
    (will search for .bed, .bim, and .fam files and return the common prefix)
    """
    bed_files = glob.glob(os.path.join(directory, "*.bed"))
    bim_files = glob.glob(os.path.join(directory, "*.bim"))
    fam_files = glob.glob(os.path.join(directory, "*.fam"))

    bed_prefixes = set([os.path.basename(bed_file).replace(".bed", "") for bed_file in bed_files])
    bim_prefixes = set([os.path.basename(bim_file).replace(".bim", "") for bim_file in bim_files])
    fam_prefixes = set([os.path.basename(fam_file).replace(".fam", "") for fam_file in fam_files])

    # Will assume that the bed prefixes are the correct ones to use in this analysis 
    # (if any bed files exist without bim/fam files, then this will throw an error)
    prefixes = bed_prefixes.intersection(bim_prefixes).intersection(fam_prefixes)

    assert len(prefixes) == len(bed_prefixes), "Some .bed files do not have corresponding .bim and .fam files! Please ensure that each PLINK binary file set has a .bed, .bim, and .fam file with the same prefix."
    
    if do_smart_sort:
        prefixes = smart_sort_filenames(prefixes)

    prefixes = [os.path.join(directory, prefix) for prefix in prefixes]

    return prefixes


def smart_sort_filenames(filenames):
    """
    Sort a list of filenames in a "smart" way, where the files are sorted by the strings and numbers in their names
    by splitting all file names into strings and numbers, in their respective orders in the name
    then sort by the strings and numbers respectively, then use the ranks to sort the files]
    """
    filename_splits = []
    for filename in filenames:
        # Split the file name into strings and numbers
        items = re.split(r'(\d+)', os.path.basename(filename))
        items = [int(item) if item.isdigit() else item for item in items]
        filename_splits.append(items)

    # Sort the files by the strings and numbers
    filenames = [x for x, _ in sorted(zip(filenames, filename_splits), key=lambda x: x[1])]

    return filenames


# Function to compute the chunksize
def compute_chunksize(n_samps, gpu_memsize_bytes, dtype_used, is_phased=False):
    """
    Computes the allowed chunk size given GPU memory, datatype, number of samples, 
    and whether the data is phased or unphased
    """

    # Calculate maximum number of elements on GPU 
    # by using the datatype size in bytes
    dtype_itemsize = np.dtype(dtype_used).itemsize
    numel_per_gpu = gpu_memsize_bytes // (dtype_itemsize)

    # We use memory totalling for R2, where:
    # chunk1 = m x n x 2, chunk2 = p x n x 2
    # where n is the number of samples, m is the number of variants in chunk1, and p is the number of variants in chunk2

    # For phased, we need:
    # 2 copies of the variant x variant matrix: 2(mp)
    # 2 copies of the 2 x variant x sample matrix: (2mn) + (2mp)
    # assume overhead of m + p (for the variant/probabilities, even though we store within the variant x sample matrix)

    # For unphased, we need (at best, in memory at once, assuming that the variant x variant matrix is larger than the variant x sample matrix):
    # 1 copy of the variant x variant matrix: (mp)
    # 1 copy of the variant x sample matrix: (2mn) - we do the summation to genotypes on the CPU
    # 1 copy of each variant matrix: (m + p) - means to center the data

    # Basically identical below, except u = 2 for phased and u = 1 for unphased
    # and v = 2 for phased and v = 1 for unphased in the formulae below
    u_val = 2 if is_phased else 1
    v_val = 2 if is_phased else 1

    # numel_per_gpu = u(mp) + (vmn) + (bnp) + (m) + (p)
    # numel_per_gpu = ump + vmn + vpn + (m + p)
    # numel_per_gpu = ump + vn(m + p) + (m + p)
    # numel_per_gpu = ump + (vn+1)(m + p)

    # solve for x assuming x=m=p

    # numel_per_gpu = ux^2 + (vn+1)(2x)
    # numel_per_gpu = ux^2 + 2(vn+1)x
    # 0 = ux^2 + 2(vn+1)x - numel_per_gpu
    
    # by using quadratic formula (always want the larger root)
    # x = (-b + sqrt(b^2 - 4ac)) / 2a
    # where a = u, b = 2(vn+1), c = -numel_per_gpu

    # Determine value of a given phased or unphased data
    # If operating on phased data, we need twice the memory for two variant x variant matrices
    a_val = u_val
    b_val = 2*(v_val*n_samps+1)
    c_val = -numel_per_gpu

    n_rows_per_rank = int((-b_val + np.sqrt(b_val**2 - 4*a_val*c_val)) // (2*a_val))


    # We are done for phased data (the above is the only-case scenario)

    # However, for unphased data, the above is the best-case scenario where 
    # the number of samples is smaller than the variant by variant matrix
    # If our computed n_rows_per_rank is smaller than the number of samples,
    # we need to recompute disregarding the variant x variant matrix 
    # and assuming we have 3 copies of variant x sample
    # as our maximum memory usage is now when 3 total strands of the variant x sample matrix
    # are in memory at once on the GPU

    if not is_phased and n_rows_per_rank < n_samps:
        # Effectively solving the above equation for x:
        # 0 = ux^2 + 2(vn+1)x - numel_per_gpu
        # but now u = 0, v = 1.5 (representing copies of the 2 x variant x sample matrix
        # but we only need three strands rather than 2 or 4, hence the 0.5)

        # 0 = (3n+2)x - numel_per_gpu
        # x = numel_per_gpu / (3n+2)

        n_rows_per_rank = int(numel_per_gpu // (3*n_samps + 2))

    return n_rows_per_rank


# Convenience MPI tags
def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
MPI_TAGS = enum('SENDDATA', 'REPORTBACK', 'SOURCESTATUS')
