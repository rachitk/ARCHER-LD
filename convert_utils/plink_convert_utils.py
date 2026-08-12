import os

import numcodecs
import zarr
import numpy as np
import bio2zarr.plink as p2z
from convert_utils.shared_convert_utils import write_concatenated_zarr, _preinit_property_dict, _property_dict_to_structured_array

import utils
from functools import partial

def plinks_to_zarr_parallel(plink_prefixes, output_zarr_folder, 
                          comm, rank, worldsize, 
                          create_combined=True, 
                          combined_prefix='combined_data', 
                          overwrite_individual=False,
                          overwrite_combined=False,
                          chunksize=2**10,
                          n_workers=0):

    # Load FAM file for first Plink file to get sample count
    fam_file = plink_prefixes[0] + ".fam"
    with open(fam_file, 'r') as f:
        sample_count = sum(1 for _ in f)

    rankprint = partial(utils.rankprint, rank=rank)

    rankprint(f"Detected {worldsize} total ranks, with {len(plink_prefixes)} PLINK files to convert (n_samples={sample_count}).", only_rank=0)

    zarr_list = get_zarr_list_from_plink(plink_prefixes, output_zarr_folder)
    combined_zarr_prefix = os.path.join(output_zarr_folder, combined_prefix)
    combined_zarr_filepath = combined_zarr_prefix + ".zarr.json"

    # If the combined file already exists, then just return that instead of re-creating it
    if(os.path.exists(combined_zarr_filepath) and not overwrite_combined and create_combined):
        rankprint(f"Combined Zarr file {combined_zarr_filepath} already exists. Pass `overwrite_combined=True` if you want to overwrite this.", only_rank=0)
        return combined_zarr_filepath

    # Given that we have multiple ranks, we'll have each rank convert one Plink file each using Bio2Zarr and then concatenate them together at the end using Kerchunk
    # We'll need at least one rank per file to convert in this fashion, but if we have more ranks than files
    if worldsize > len(plink_prefixes):
        rankprint(f"More ranks ({worldsize}) than PLINK files ({len(plink_prefixes)}), some ranks will be idle during initial conversion.", only_rank=0)

    # Assign files to ranks in a round-robin fashion (so that if we have more ranks than files, the extra ranks will just be idle)
    rank_plinks = {i: plink_prefixes[i] for i in range(len(plink_prefixes)) if i % worldsize == rank}
    rankprint(f"Assigned PLINK files ({len(rank_plinks)}) for conversion for this rank: {list(rank_plinks.values())}")

    # Each rank converts its assigned files to VCF-Zarr and then converts to our specification
    # Note, the VCF-Zarr spec has almost all of our fields, but is missing a few here or there
    for plink_i, plink_prefix in rank_plinks.items():
        rankprint(f"Converting {plink_prefix} to Zarr...")

        vcfzarr_filename = os.path.basename(plink_prefix) + ".vcz"
        vcfzarr_filepath = os.path.join(output_zarr_folder, vcfzarr_filename)

        skip_convert = False

        if os.path.exists(vcfzarr_filepath) and not overwrite_individual:
            vcfzarr = zarr.open(vcfzarr_filepath, mode='r')

            try:
                gt_array = vcfzarr['call_genotype']
                n_variants = gt_array.shape[0]
                rankprint(f"File {vcfzarr_filepath} already exists and overwrite_individual is False, skipping VCF-Zarr conversion for this file. Detected {n_variants} variants in existing file.")
                skip_convert = True
            except Exception as e:
                rankprint(f"File {vcfzarr_filepath} already exists and overwrite_individual is False, but could not be read properly, likely due to being a corrupt file from a previous failed conversion attempt (or this is a worker being initialized for the first attempt). Re-converting this file.")
                # Delete the file
                os.remove(vcfzarr_filepath)
                skip_convert = False

        if not skip_convert:
            vcfzarr = p2z.convert(plink_prefix, vcfzarr_filepath, 
                                variants_chunk_size=chunksize, 
                                samples_chunk_size=sample_count,
                                worker_processes=n_workers,
                                show_progress=False) # only show progress if we're running in serial to avoid cluttering the output
 
            rankprint(f"Converted {plink_prefix} to VCF-Zarr spec with bio2zarr.")

        out_zarr_filepath = zarr_list[plink_i]

        if os.path.exists(out_zarr_filepath) and not overwrite_individual:
            rankprint(f"File {out_zarr_filepath} already exists and overwrite_individual is False, skipping conversion to final Zarr spec for this file.")
        else:
            convert_vcfzarr_spec(vcfzarr, out_zarr_filepath, verbose=True, rank=rank) 

            rankprint(f"Finished converting {plink_prefix} to our Zarr spec.")


    rankprint(f"This rank finished converting assigned PLINK files to Zarr spec, waiting for all ranks to finish before combining files...")

    # Wait until all ranks are done before continuing after this point
    # (so that we can guarantee the files are all done before combining)
    comm.Barrier()

    # If we're not combining, return the list of Zarr files
    if(not create_combined):
        return zarr_list
    

    # Otherwise, making the combined Zarr file will always be the job
    # of rank 0 to avoid multiple ranks trying to write to the same file at once
    if(rank == 0):
        rankprint("Combining Zarr files...")

        # Write the combined Zarr reference file
        # This will be a JSON reference to the individual Zarr files
        combined_zarr_filepath = write_concatenated_zarr(zarr_list, combined_zarr_filepath, do_virtual=True)

        rankprint(f"Combination complete ({combined_zarr_filepath}).")
        rankprint(f"Waiting for all ranks to finish before returning.")

    # Wait until all ranks are done before returning (as otherwise the combined file may not be ready)
    comm.Barrier()

    return combined_zarr_filepath


def plinks_to_zarr(plink_prefixes, output_folder, create_combined=True,
                               combined_prefix='combined_data', 
                               overwrite_individual=False,
                               overwrite_combined=False,
                               chunksize=2**10,
                               comm=None, rank=0, worldsize=1, 
                               parallel=False,
                               n_workers=0):
    """
    Converts a list of Plink files to a genotype matrix file in a specified format.
    Will make an output file for each Plink file and then either return a list of the files

    OR, if create_combined is True, will concatenate them all together 
    and then return the path to the concatenated file.
    """

    # TODO: make this blocking for each rank and only use rank 0 for making the folder
    os.makedirs(output_folder, exist_ok=True)

    if(isinstance(plink_prefixes, str)):
        plink_prefixes = [plink_prefixes]


    # In this case, Plink conversions are best done across multiple ranks unless there is only one allowed
    # as bio2zarr only converts one file at a time 
    if(worldsize == 1 or not parallel):
        if worldsize > 1:
            print("Multiple processes detected but parallel conversion is disabled, converting PLINK files to genotype matrices in serial...")
        else:
            print("Detected only one process, converting PLINK files to genotype matrices in serial...")

        if rank == 0:
            # Just run the parallel version but with worldsize=1 to do the conversion in serial on one rank, and then return the combined file if requested
            ret_zarr =  plinks_to_zarr_parallel(plink_prefixes, output_folder, 
                                        comm, rank, worldsize=1,
                                        create_combined=create_combined,
                                        combined_prefix=combined_prefix, 
                                        overwrite_individual=overwrite_individual, 
                                        overwrite_combined=overwrite_combined,
                                        chunksize=chunksize,
                                        n_workers=n_workers)
        else:
            if create_combined:
                ret_zarr = os.path.join(output_folder, combined_prefix + ".zarr.json")
            else:
                ret_zarr = get_zarr_list_from_plink(plink_prefixes, output_folder)

        comm.Barrier()
        return ret_zarr

    else:
        return plinks_to_zarr_parallel(plink_prefixes, output_folder, 
                                        comm, rank, worldsize,
                                        create_combined=create_combined,
                                        combined_prefix=combined_prefix, 
                                        overwrite_individual=overwrite_individual, 
                                        overwrite_combined=overwrite_combined,
                                        chunksize=chunksize, 
                                        n_workers=n_workers)
    


def get_zarr_list_from_plink(plink_prefixes, output_folder):
    """
    Helper function to get the list of Zarr files that would be generated from a list of Plink files
    (used for testing and for the case where we want to convert but not combine)
    """
    zarr_list = []
    for plink_prefix in plink_prefixes:
        zarr_filename = os.path.basename(plink_prefix) + ".zarr"
        zarr_filepath = os.path.join(output_folder, zarr_filename)

        zarr_list.append(zarr_filepath)

    return zarr_list



def convert_vcfzarr_spec(input_vcfzarr, output_zarr_filepath, verbose=False, rank=None):
    """
    Converts a VCF-Zarr file in the specification used by Bio2Zarr to the specification used by our LD GPU pipeline.
    This is a helper function for plinks_to_zarr that does the actual conversion of the VCF-Zarr spec to our spec after Bio2Zarr has done its conversion from PLINK to VCF-Zarr.
    NOTE: we'll eventually just transition almost entirely to the VCF-Zarr spec for our own work, so this will eventually shift to a function that adds metadata fields that we want (compliant with spec)
    """
    # input_vcfzarr is a loaded Zarr group from the VCF-Zarr spec
    # output_zarr_filepath is the path to the output Zarr file in our spec that we want to create

    if rank is not None:
        rankprint = partial(utils.rankprint, rank=rank)
    else:
        rankprint = print

    # Get some basic important information from the input VCF-Zarr file that we'll need for the conversion
    gt_array = input_vcfzarr['call_genotype']
    n_variants = gt_array.shape[0]
    n_samples = gt_array.shape[1]
    chunkshape = gt_array.chunks

    chunksize = chunkshape[0]

    # Open the output Zarr
    vcf_zarr_g = zarr.open(output_zarr_filepath, mode='w', zarr_format=2)

    # Copy over the relevant arrays from the VCF-Zarr spec to our spec
    # This is mostly just renaming and reformatting, but also includes some minor transformations

    # For example, we need to pad so that the final chunk is a multiple of the chunk size,
    # since Kerchunk requires this for concatenating the final array views
    offset = 0
    if n_variants % chunksize != 0:
        offset = chunksize - (n_variants % chunksize)

    n_variants_with_padding = n_variants + offset

    # Start with the genotype array, which is a straight copy though we can't load it all into memory
    # so we have to do it in chunks instead
    zarr_data_gt = vcf_zarr_g.create('gt', shape=(n_variants_with_padding, n_samples, 2),
                                        chunks=chunkshape, dtype=np.int8, 
                                        fill_value=-1,
                                        config={'write_empty_chunks': False})
    

    # Then make the metadata arrays, assign, and save to the Zarr file
    # Most we can do in memory as they are small enough, and we can assign them all at once at the end after creating the arrays with the correct shapes and dtypes
    zarr_metadata = _property_dict_to_structured_array(_preinit_property_dict(n_variants_with_padding))

    # Copy the metadata that can be straight copied without conversions (since they're small)
    # as described in the copy_dict below
    copy_dict = {
        'variant_position': 'POS',
        'variant_id': 'ID',
    }

    zarr_metadata = vcfzarr_to_zarr_meta_copy(input_vcfzarr, zarr_metadata, copy_dict=copy_dict, n_variants=n_variants)

    # Then handle slightly more complex metadata that needs some conversion/work before it can be copied over, but can still be done in memory since these are small arrays
    # CHROM = variant_contig indexes into contig_id
    zarr_metadata['CHROM'][:n_variants] = input_vcfzarr['contig_id'][input_vcfzarr['variant_contig']]
    # REF and ALT are stored in the variant_allele array, where the first allele is the REF and the rest are the ALTs, so we can extract those and format them as needed
    zarr_metadata['REF'][:n_variants] = input_vcfzarr['variant_allele'][:,0]
    zarr_metadata['ALT'][:n_variants] = np.apply_along_axis('/'.join, 1, input_vcfzarr['variant_allele'][:,1:]) # TODO: check if this leads to the same result as in the VCF conversion script
    # N_ALLELES is just the number of alleles for each variant, which is the count of non-empty strings in the variant_allele array for each variant
    zarr_metadata['N_ALLELES'][:n_variants] = np.sum(input_vcfzarr['variant_allele'][:,1:] != '', axis=1)
    # FILTER is not pulled from Plink files, so we just leave it alone (as fill)


    # Now copy over the genotype in chunks (TODO: straight copy without chunking or create virtual reference to the original array if the chunking is the same - will still need to load the genotypes but won't need to rewrite to disk)
    # as well as any metadata that needs to be computed from those chunks (which may not necessarily be genotypes, like the phase data)
    for i in range(0, n_variants, chunksize):
        i_capped = min(i+chunksize, n_variants)

        # Copy GT array
        gt_chunk = gt_array[i:i_capped]
        zarr_data_gt[i:i_capped] = gt_chunk

        # AAF needs to be computed from the GT array chunk, aggregated over all possible alternate alleles
        # We can compute this from the GT chunk we already have in memory, so we don't need to load anything extra
        alt_allele_bool = gt_chunk > 0 # shape (chunk_size, n_samples, 2), bool array where True indicates presence of an alt allele
        alt_allele_count = np.sum(alt_allele_bool, axis=(1,2)) # shape (chunk_size,), counts of alt alleles per variant in the chunk
        called_bool = gt_chunk > -1
        total_allele_count = np.sum(called_bool, axis=(1,2)) # shape (chunk_size,), counts of total alleles (including ref and alt) per variant in the chunk
        total_allele_count = np.where(total_allele_count == 0, 1, total_allele_count) # to avoid division by zero, if there are no called alleles, we set the total allele count to 1 so that the AAF is set to 0 (since alt_allele_count will also be 0 in this case)
        zarr_metadata['AAF'][i:i_capped] = alt_allele_count / total_allele_count

        # N_CALLED is the number of samples where either allele is called (i.e. not -1) for each variant, so we can compute this from the GT chunk as well
        any_called_bool = np.any(called_bool, axis=-1) # shape (chunk_size, n_samples), bool array where True indicates either allele is called for a sample at a variant
        n_called_chunk = np.sum(any_called_bool, axis=1) # shape (chunk_size,), counts of samples with at least one called allele per variant
        zarr_metadata['N_CALLED'][i:i_capped] = n_called_chunk

        # N_UNKNOWN is number of samples where neither allele is called
        # this is just the number of samples - n_called_chunk
        n_unknown_chunk = n_samples - n_called_chunk
        zarr_metadata['N_UNKNOWN'][i:i_capped] = n_unknown_chunk

        # CALL_RATE needs to be computed from the GT array chunk as well, as the proportion of called genotypes (i.e. not -1) across all samples for each variant
        # this is just total_allele_count from above divided by (2 * number of samples) since each sample has 2 alleles
        zarr_metadata['CALL_RATE'][i:i_capped] = total_allele_count / (2 * n_samples)

        # N_HOMREF is the count of genotypes where both alleles are 0 (homozygous reference)
        ref_bool = gt_chunk == 0 # shape (chunk_size, n_samples, 2), bool array where True indicates a ref genotype
        homref_bool = np.all(ref_bool, axis=-1) # shape (chunk_size, n_samples), bool array where True indicates a homozygous reference genotype
        n_homref_chunk = np.sum(homref_bool, axis=1)
        zarr_metadata['N_HOMREF'][i:i_capped] = n_homref_chunk

        # N_HOMALT is the count of genotypes where both alleles are > 0 (homozygous alternate)
        # (and we already have the alt allele bool from above, so we can just use that)
        homalt_bool = np.all(alt_allele_bool, axis=-1) # shape (chunk_size, n_samples), bool array where True indicates a homozygous alternate genotype
        n_homalt_chunk = np.sum(homalt_bool, axis=1)
        zarr_metadata['N_HOMALT'][i:i_capped] = n_homalt_chunk

        # N_HET is the count of genotypes where one allele is 0 and the other is > 0 (heterozygous)
        het_bool = np.logical_and(np.any(ref_bool, axis=-1), np.any(alt_allele_bool, axis=-1)) # shape (chunk_size, n_samples), bool array where True indicates a heterozygous genotype
        n_het_chunk = np.sum(het_bool, axis=1)
        zarr_metadata['N_HET'][i:i_capped] = n_het_chunk

        # N_PHASED needs to be computed from input_vcfzarr['call_genotype_phased']
        phased_chunk = input_vcfzarr['call_genotype_phased'][i:i_capped]
        n_phased_chunk = np.sum(phased_chunk, axis=-1)
        zarr_metadata['N_PHASED'][i:i_capped] = n_phased_chunk

        if verbose:
            rankprint(f"Converted {i_capped}/{n_variants} variants...")
    

    # Actually store the metadata array in the Zarr file now that we've filled it in
    zarr_data_meta = vcf_zarr_g.create_array('meta', data=zarr_metadata,
                                        chunks=(chunksize,))

    return output_zarr_filepath



def vcfzarr_to_zarr_meta_copy(input_vcfzarr, zarr_metadata,
                              copy_dict={}, n_variants=None):
    """
    Straight copy metadata for those metadata keys that need no modification or changing at all
    (e.g. variant_position -> POS, variant_id -> ID, etc.), on a per-variant basis

    Note that if the output array (the zarr metadata) is longer than the input array, only the 
    number of elements that match will be assigned (this can happen due to padding) 
    """

    # Compute the number of variants in the real data if not provided, 
    # as the length of the variant_id array in the input VCF-Zarr file 
    # (or any other array that has one element per variant, since they should all have the same length)
    if n_variants is None:
        n_variants = input_vcfzarr['call_genotype'].shape[0]

    for in_key, out_key in copy_dict.items():
        zarr_metadata[out_key][:n_variants] = input_vcfzarr[in_key]

    return zarr_metadata