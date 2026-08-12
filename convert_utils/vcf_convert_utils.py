import os
import time

import cyvcf2
import numpy as np # maybe cuda/cupy?
import zarr
import numcodecs
import itertools

from convert_utils.shared_convert_utils import write_concatenated_zarr, _preinit_property_dict, _property_dict_to_structured_array

# Note: we considered using Bio2zarr's vcf2zarr and just going with their spec, but we found that the speed of their conversion 
# is unfortunately still slower as it appears they serialize the file read with the line processing, while we split the two apart
# allowing for multiple processes to process the lines while the file is being read by a single process

# TODO: convert our results to match the VCF-Zarr spec instead of the other way around
# https://github.com/sgkit-dev/vcf-zarr-spec/blob/main/vcf_zarr_spec.md
# Should be fairly easy, though we don't implement some of the specific fields they do and have a few extras, we implement most


def vcfs_to_zarr_parallel(vcfs, output_zarr_folder, 
                          comm, rank, worldsize, 
                          create_combined=True, 
                          combined_prefix='combined_data', 
                          overwrite_individual=False,
                          overwrite_combined=False,
                          chunksize=2**10,
                          multivcf_parallel=True):
  
    from mpi4py import MPI
    import utils
    from utils import MPI_TAGS
    from functools import partial
    import tempfile

    zarr_list = []
    status = MPI.Status()

    rankprint = partial(utils.rankprint, rank=rank)

    combined_zarr_prefix = os.path.join(output_zarr_folder, combined_prefix)
    combined_zarr_filepath = combined_zarr_prefix + ".zarr.json"

    # If the combined file already exists, then just return that instead of re-creating it
    if(os.path.exists(combined_zarr_filepath) and not overwrite_combined and create_combined):
        rankprint(f"Combined Zarr file {combined_zarr_filepath} already exists. Pass `overwrite_combined=True` if you want to overwrite this.")
        return combined_zarr_filepath


    # One process loading each VCF instead of all loading all VCFs
    # and then sending messages with tags to the other processes
    if(multivcf_parallel):
        if(worldsize < (len(vcfs)*2)):
            raise ValueError("Number of processes must be greater than twice the number of VCFs to use multivcf_parallel=True (must have one reading process and at least one writing process per VCF)")
            # Note that if this isn't true, then there isn't a good reason to use multivcf_parallel=True and we should just use the serial approach
            # since we cannot parallelize reading the same VCF file and source ranks do not become workers upon finishing

        vcf_ranks = list(range(0, len(vcfs)))
        rem_ranks = list(range(len(vcfs), worldsize))

        worker_ranks_per_vcf = np.array_split(rem_ranks, len(vcfs))
        worker_to_source_dict = {worker: source for source, workers in enumerate(worker_ranks_per_vcf) for worker in workers}

    else:
        vcf_ranks = [0] * len(vcfs)
        rem_ranks = list(range(1, worldsize))
        worker_ranks_per_vcf = [list(range(1, worldsize))] * len(vcfs)
        worker_to_source_dict = {worker: 0 for worker in rem_ranks}

    
    IS_SOURCE = rank in vcf_ranks
    IS_WORKER = rank in rem_ranks

    if(IS_SOURCE and IS_WORKER):
        raise ValueError(f"DEBUG: Rank {rank} is currently both a source and a worker. This should not happen - please debug.")
    
    if(IS_SOURCE):
        rankprint(f"Initialized as a source for VCF {vcfs[rank]}.")
        remaining_valid_source_ranks = np.unique(vcf_ranks).tolist()

    for vcf_i, vcf in enumerate(vcfs):
        # Bit hacky, but people can pass in GZipped VCFs and so on, and 
        # people in genetics often add random additional "." in their files
        # so we want to remove the last instance of ".vcf" and all characters after that
        vcf_cleanname = os.path.basename(vcf)
        vcf_cleanname = vcf_cleanname[:vcf_cleanname.rfind(".vcf")]
        chunk_subdir = os.path.join(output_zarr_folder, f"{vcf_cleanname}_chunks")
        output_zarr_vcf_prefix = os.path.join(output_zarr_folder, vcf_cleanname)
        output_zarr_vcf = output_zarr_vcf_prefix + '.zarr'

        zarr_list.append(output_zarr_vcf)

        vcf_rank = vcf_ranks[vcf_i]
        vcf_worker_ranks = list(worker_ranks_per_vcf[vcf_i])
        

        if(rank == vcf_rank):
            # Basic checks:
            if not os.path.exists(vcf):
                raise FileNotFoundError(f"VCF file {vcf} not found")
            
            if os.path.exists(output_zarr_vcf) and not overwrite_individual:
                rankprint(f"Converted file {output_zarr_vcf} already exists. Pass `overwrite_individual=True` if you want to overwrite this.")

                # Receive all initial messages from the workers 
                # (so that they are now in the pool of available workers)
                for worker_i in vcf_worker_ranks:
                    worker_data = comm.recv(source=worker_i, status=status, tag=MPI_TAGS.REPORTBACK)
                    # rankprint(f"Got initial message from {worker_i} to ignore.")

            else:
                # Open the VCF, read chunksize lines, and pass to a free process
                # to convert to Zarr iteratively
                # TODO: this is the current rate-limiting step, so we may want to
                # figure out ways to speed this up (e.g. by using htslib directly)

                os.makedirs(chunk_subdir, exist_ok=True)
                is_gz = vcf.endswith("gz")
                header_char = '#'
                delim_char = '\t'
                str_type = 's'

                if(is_gz):
                    import gzip
                    vcf_f = gzip.open(vcf,'r')
                    header_char = header_char.encode('utf-8')
                    delim_char = delim_char.encode('utf-8')
                    str_type = 'b'
                else:
                    vcf_f = open(vcf, "r")

                header_lines = []

                # Read the various header lines (TODO: prune unneeded ones?)
                # until we reach the ACTUAL header line (last one with a '#' at the start)
                for line in vcf_f:
                    if line.startswith(header_char):
                        header_lines += [line]
                    else:
                        # Seek back to the start of the last line
                        vcf_f.seek(vcf_f.tell() - len(line))
                        break

                # Get the number of samples for this VCF
                samples = header_lines[-1].split(delim_char)[9:]
                n_samples = len(samples)

                # Loop lines and send to a process to convert to Zarr
                variant_i = 0
                chunk_i = 0
                chunk_file_list = []

                rankprint(f"Converting {vcf} to Zarr chunks...")

                while True:
                    chunk_lines = list(itertools.islice(vcf_f, chunksize))

                    if not chunk_lines:
                        break

                    # Queue a receive for a free worker rank from all of the workers
                    worker_data = comm.recv(source=MPI.ANY_SOURCE, status=status, tag=MPI_TAGS.REPORTBACK)
                    worker_i = status.Get_source()

                    # Add this worker to this source's pool of available workers if it is not already there
                    # indicated by the worker sending something other than None to this source
                    # (we do "ACQUIRE_SOURCE" to indicate that we are ready to receive data)
                    if(worker_data is not None):
                        rankprint(f"Added {worker_i} to source {rank}.")
                        vcf_worker_ranks.append(worker_i)
                    
                    # Send the lines to the worker rank (and the number of samples)
                    comm.send((chunk_i, header_lines, chunk_lines, len(chunk_lines), n_samples, str_type, chunk_subdir), dest=worker_i, tag=MPI_TAGS.SENDDATA)

                    rankprint(f"Sent VCF({vcf_i}) line-chunk {chunk_i} to rank {worker_i} (variant index: {variant_i})")

                    # Store chunk name
                    chunk_file_list.append(os.path.join(chunk_subdir, f"{chunk_i}.zarr"))

                    variant_i += len(chunk_lines)
                    chunk_i += 1

                # Close the file once we're done iterating over it
                vcf_f.close()


                # Receive all remaining messages from the workers (ensures they are done)
                for worker_i in vcf_worker_ranks:
                    worker_data = comm.recv(source=worker_i, status=status, tag=MPI_TAGS.REPORTBACK)

                rankprint(f"Writing concatenated VCF {vcf} to Zarr.")

                # Create the final combined Zarr file for this VCF
                # in the path stored in `output_zarr_vcf` (has the .zarr suffix - do_virtual is False)
                write_concatenated_zarr(chunk_file_list, output_zarr_vcf, do_virtual=False)

            rankprint(f"Finished converting VCF {vcf} to Zarr.")

            if(multivcf_parallel):
                # Notify all the other source ranks that we are done and
                # figure out what sources are done to have a list of available sources
                # (this is a bit hacky and could be done better)
                remaining_valid_source_ranks.remove(rank)
                current_remaining_source_ranks = remaining_valid_source_ranks.copy()
                rankprint(f"Remaining possible ranks: {current_remaining_source_ranks}")

                for source_rank in current_remaining_source_ranks:
                    comm.isend("done", dest=source_rank, tag=MPI_TAGS.SOURCESTATUS)
                
                time.sleep(10)

                for source_rank in current_remaining_source_ranks:
                    test_val = comm.iprobe(source=source_rank, tag=MPI_TAGS.SOURCESTATUS)

                    time.sleep(1)
                    
                    if(test_val):
                        rankstatus = comm.recv(source=source_rank, tag=MPI_TAGS.SOURCESTATUS)
                    else:
                        rankstatus = "assumed working"

                    rankprint(f"Final status from source rank {source_rank}: {rankstatus}")

                    if(rankstatus == "done"):
                        rankprint(f"Source rank {source_rank} is done.")
                        remaining_valid_source_ranks.remove(source_rank)


                # If we reallocated the workers, then we weren't the last process to finish
                did_realloc = False

                if(len(remaining_valid_source_ranks) != 0):
                    did_realloc = True

                    # Reallocate workers to the remaining ranks if we are doing multi-VCF parallelism
                    worker_ranks_reallocate = np.array_split(vcf_worker_ranks, len(remaining_valid_source_ranks))
                    worker_to_new_source_dict = {worker: remaining_valid_source_ranks[i] for i, workers in enumerate(worker_ranks_reallocate) for worker in workers}

                    new_worker_ranks = vcf_worker_ranks.copy()

                    # Tell all of the workers to go to their new source
                    for worker_i in vcf_worker_ranks:
                        new_source = worker_to_new_source_dict[worker_i]
                        rankprint(f"Reallocating worker {worker_i} to source {new_source}.")
                        comm.send((None, new_source, new_source, None, None, None, None), dest=worker_i, tag=MPI_TAGS.SENDDATA)

                        # Remove the worker from the old source's list of workers
                        new_worker_ranks.remove(worker_i)

                    vcf_worker_ranks = new_worker_ranks


                # Continually check the remaining sources if they are done
                # Once all sources are done, we can exit
                # We can do this in a blocking way because we no longer care about the world
                # and only care about whether the other sources are done
                while True:
                    current_remaining_source_ranks = remaining_valid_source_ranks.copy()

                    for source_rank in current_remaining_source_ranks:
                        rankstatus = comm.recv(source=source_rank, tag=MPI_TAGS.SOURCESTATUS)

                        if(rankstatus == "done"):
                            remaining_valid_source_ranks.remove(source_rank)
                            rankprint(f"Source rank {source_rank} is done.")

                    if len(remaining_valid_source_ranks) == 0:
                        rankprint(f"All sources are done. Waiting 10 seconds before continuing...")

                        time.sleep(10)

                        # (Try to) tell the workers that they are now fully done
                        # We'll allow for a 10 second wait for them to respond before
                        # giving up and moving on (we'll cancel them)
                        # only if this was the LAST process to finish as it'll have all the workers
                        # (otherwise, we'll just continually wait for the last process to finish)
                        
                        # First, check and see if this process has received new workers
                        # and if so, add them to the list of workers to kill
                        # Queue a receive for a free worker rank from all of the workers
                        rankprint(f"Checking all ranks to see if there are workers waiting...")
                        for possible_worker_i in range(worldsize):
                            test_newworker = comm.iprobe(source=possible_worker_i, tag=MPI_TAGS.REPORTBACK)
                            time.sleep(0.1)

                            if(test_newworker):
                                worker_data = comm.recv(source=possible_worker_i, status=status, tag=MPI_TAGS.REPORTBACK)
                                worker_i = status.Get_source()

                                # Add this worker to this source's pool of available workers if it is not already there
                                # indicated by the worker sending something other than None to this source
                                # (we do "ACQUIRE_SOURCE" to indicate that we are ready to receive data)
                                if(worker_data is not None):
                                    vcf_worker_ranks.append(worker_i)

                        worker_end_reqs = []
                        for worker_i in vcf_worker_ranks:
                            end_req = comm.isend((None, None, None, None, None, None, None), dest=worker_i, tag=MPI_TAGS.SENDDATA)
                            worker_end_reqs.append(end_req)

                        rankprint(f"Sent signal to all assigned workers {vcf_worker_ranks} to stop. Waiting 10 seconds.")

                        time.sleep(10)
                        for worker_end_req in worker_end_reqs:
                            if not worker_end_req.test():
                                worker_end_req.cancel()

                        break

            else:
                # We were working serially, so we are done at the last VCF
                if vcf_i == len(vcfs) - 1:
                    # We are done with the last VCF
                    # Send a signal to all workers to stop
                    for worker_i in vcf_worker_ranks:
                        comm.send((None, None, None, None, None, None, None), dest=worker_i, tag=MPI_TAGS.SENDDATA)



    if IS_WORKER:
        worker_source = worker_to_source_dict[rank]
        # rankprint(f"Initialized as a worker for VCF {vcfs[worker_source]}.")

        # Send a starting signal to the source rank
        comm.send(None, dest=worker_source, tag=MPI_TAGS.REPORTBACK)

        # Receive initial data
        chunk_i, header_lines, chunk_lines, n_vars, n_samples, str_type, chunk_subdir = comm.recv(source=worker_source, status=status, tag=MPI_TAGS.SENDDATA)


        while True:

            if(chunk_i is None):
                # Check if we actually got a notification to switch sources
                # in which case, header_lines and chunk_lines won't be None
                # and will instead both be the new source ID to swap to and from which we should
                # queue a new receive (this is hacky, and there's a better way to do this, but this works)                
                if(header_lines is not None and chunk_lines is not None):
                    worker_source = header_lines
                    rankprint(f"Switching worker to new source rank {worker_source}.")
                    comm.send("ACQUIRE_SOURCE", dest=worker_source, tag=MPI_TAGS.REPORTBACK)
                    chunk_i, header_lines, chunk_lines, n_vars, n_samples, str_type, chunk_subdir = comm.recv(source=worker_source, status=status, tag=MPI_TAGS.SENDDATA)
                    continue
                else:
                    # We've received the final chunk and are done
                    break

            # rankprint(f"Received chunk {chunk_i} data with {n_vars} variants from rank {worker_source}.")
            chunk_zarr_file = os.path.join(chunk_subdir, f"{chunk_i}.zarr")

            # Parse the header and chunk lines into a single byte-string 
            # (Note: may already be byte, so we check)
            cat_str = '' if str_type == 's' else b''
            chunk_str = cat_str.join(header_lines + chunk_lines)
            chunk_str = chunk_str.encode() if str_type == 's' else chunk_str


            if(os.path.exists(chunk_zarr_file) and not overwrite_individual):
                rankprint(f"Chunk {chunk_zarr_file} already exists. Skipping.")
                comm.send(None, dest=worker_source, tag=MPI_TAGS.REPORTBACK)
                chunk_i, header_lines, chunk_lines, n_vars, n_samples, str_type, chunk_subdir = comm.recv(source=worker_source, status=status, tag=MPI_TAGS.SENDDATA)
                continue

            # Worker converts strings to array (via cyvcf2) and then to Zarr
            # TODO: call the htslib functions directly to avoid the multiple layers of overhead
            # in terms of writing a temporary file, forcing a read with CyVCF2, etc.
            # NOTE: CyVCF2 can read from stdin, so consider the following solution:
            # https://github.com/brentp/cyvcf2/issues/47
            # though multiple processes reading/writing from/to the same stdin
            # may be an issue (do they share a stdin?)
            chunk_gt_data = np.empty((n_vars, n_samples, 2), dtype=np.int8)
            chunk_vardata = _preinit_property_dict(chunksize)
            chunk_vardata = _property_dict_to_structured_array(chunk_vardata)

            with tempfile.NamedTemporaryFile(dir=chunk_subdir, delete=True) as vcf_tp:
                vcf_tp.write(chunk_str)
                vcf_tp.flush()

                vcf_f = cyvcf2.VCF(vcf_tp.name, gts012=True, strict_gt=True)

                for var_i, variant in enumerate(vcf_f):
                    # Get "phased" data and phasing info
                    phased_gt_data = np.array(variant.genotypes, dtype=np.int8)
                    gt_phase_info = phased_gt_data[:, 2].astype(bool)
                    phased_gt_data = phased_gt_data[:, :2]

                    # Force all unphased, heterozygous data to be 1/0 on the second axis
                    # (as we are going to coerce the heterozygotes to all be on the "same stand")
                    unphased_mask = ~gt_phase_info
                    het_mask = phased_gt_data.sum(axis=1) == 1
                    phased_gt_data[het_mask & unphased_mask, :] = np.array([1, 0], dtype=np.int8)

                    chunk_gt_data[var_i] = phased_gt_data
                    variant_props = _get_variant_properties(variant)

                    for k,v in variant_props.items():
                        chunk_vardata[k][var_i] = v


            # Convert to Zarr (make a smaller Zarr file for each chunk)
            chunk_zarr = zarr.open(chunk_zarr_file, mode='w', zarr_format=2)

            # Write the "phased" data to 'gt' array in Zarr group
            chunk_zarr_data = chunk_zarr.create_array('gt', data=chunk_gt_data, 
                                               chunks=(chunksize, n_samples, 2),
                                               fill_value=-1,
                                               config={'write_empty_chunks': False})

            # Write the metadata to the Zarr file
            chunk_zarr_meta = chunk_zarr.create_array('meta', data=chunk_vardata, 
                                               chunks=(chunksize,))

            rankprint(f"Saved chunk {chunk_i} data into {chunk_zarr_file}.")

            # Send a signal to the source rank that we are ready to receive data
            comm.send(None, dest=worker_source, tag=MPI_TAGS.REPORTBACK)

            # Receive data on other ranks
            chunk_i, header_lines, chunk_lines, n_vars, n_samples, str_type, chunk_subdir = comm.recv(source=worker_source, status=status, tag=MPI_TAGS.SENDDATA)
                

    # Wait until all ranks are done before continuing after this point
    # (so that we can guarantee the files are all done before combining)
    comm.Barrier()

    # If we're not combining, return the list of Zarr files
    if(not create_combined):
        return zarr_list

    # Otherwise, making the combined Zarr file will always be the job
    # of rank 0 (which is also the source rank if there is only one VCF)
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


def vcfs_to_zarr_serial(vcfs, output_zarr_folder, create_combined=True,
                                 combined_prefix='combined_data', 
                                 overwrite_individual=False,
                                 overwrite_combined=False,
                                 chunksize=2**10):
    """
    Converts a list of VCF files to a genotype matrix file in Zarr format (serially).
    Will make an Zarr file for each VCF file and then either return a list of the files

    OR, if create_combined is True, will concatenate them all together 
    and then return the path to the concatenated file.
    """

    zarr_list = []

    combined_zarr_prefix = os.path.join(output_zarr_folder, combined_prefix)
    combined_zarr_filepath = combined_zarr_prefix + ".zarr.json"


    # If the combined file already exists, then just return that instead of re-creating it
    if(os.path.exists(combined_zarr_filepath) and not overwrite_combined and create_combined):
        print(f"Combined Zarr file {combined_zarr_filepath} already exists. Pass `overwrite_combined=True` if you want to overwrite this.")
        return combined_zarr_filepath

    # For each VCF, load and convert
    for vcf in vcfs:
        # Bit hacky, but people can pass in GZipped VCFs and so on, and 
        # people in genetics often add random additional "." in their files
        # so we want to remove the last instance of ".vcf" and all characters after that
        vcf_cleanname = os.path.basename(vcf)
        vcf_cleanname = vcf_cleanname[:vcf_cleanname.rfind(".vcf")]
        output_zarr_vcf = os.path.join(output_zarr_folder, f"{vcf_cleanname}.zarr")

        zarr_list.append(output_zarr_vcf)

        if not os.path.exists(vcf):
            raise FileNotFoundError(f"VCF file {vcf} not found")
        
        if os.path.exists(output_zarr_vcf) and not overwrite_individual:
            print(f"Converted file {output_zarr_vcf} already exists. Pass `overwrite_individual=True` if you want to overwrite this.")
        else:
            # Time the appending
            start_time = time.time()
            orig_start_time = start_time
            print(f"Starting conversion of {vcf} to Zarr...")
            overall_var_i = 0

            # Load the VCF in chunks and convert to Zarr
            vcf_f = cyvcf2.VCF(vcf, gts012=True, strict_gt=True)
            n_samples = len(vcf_f.samples)

            # Initialize the chunked data
            chunk_gt_data = np.empty((chunksize, n_samples, 2), dtype=np.int8)
            chunk_vardata = _property_dict_to_structured_array(_preinit_property_dict(chunksize))

            # Convert to Zarr (make a smaller Zarr file for each chunk)
            vcf_zarr_g = zarr.open(output_zarr_vcf, mode='w', zarr_format=2)

            # Make an array for the genotype data
            vcf_zarr = vcf_zarr_g.create('gt', shape=(0, n_samples, 2),
                                         chunks=(chunksize, n_samples, 2), dtype=np.int8, 
                                         fill_value=-1,
                                         config={'write_empty_chunks': False})

            # Make an array for the metadata
            vcf_zarr_meta = vcf_zarr_g.create('meta', shape=(0,),
                                              dtype=chunk_vardata.dtype, 
                                              chunks=(chunksize,))
            
            chunk_i = 0

            for variant in vcf_f:
                # Get "phased" data and phasing info
                phased_gt_data = np.array(variant.genotypes, dtype=np.int8)
                gt_phase_info = phased_gt_data[:, 2].astype(bool)
                phased_gt_data = phased_gt_data[:, :2]

                # Force all unphased, heterozygous data to be 1/0 on the second axis
                # (as we are going to coerce the heterozygotes to all be on the "same stand")
                unphased_mask = ~gt_phase_info
                het_mask = phased_gt_data.sum(axis=1) == 1
                phased_gt_data[het_mask & unphased_mask, :] = np.array([1, 0], dtype=np.int8)

                chunk_gt_data[chunk_i] = phased_gt_data
                variant_props = _get_variant_properties(variant)

                for k,v in variant_props.items():
                    chunk_vardata[k][chunk_i] = v

                chunk_i += 1
                overall_var_i += 1

                if(chunk_i == chunksize):
                    vcf_zarr.append(chunk_gt_data)
                    vcf_zarr_meta.append(chunk_vardata)
                    chunk_i = 0

                    # and reset the chunk_vardata dict
                    chunk_vardata = _property_dict_to_structured_array(_preinit_property_dict(chunksize))

                    chunk_time = time.time()
                    print(f"({overall_var_i}): Chunk time: {chunk_time - start_time}")
                    start_time = chunk_time


            # In case the last chunk is not full
            # We need to pad the variant axis to be a multiple of the chunksize
            # (since Zarr doesn't currently support concatenation of arrays where the last chunk is a different size)
            # (this is a bit hacky and will lead to some wasted space)
            # TODO: fix this when ZEP3 is implemented (https://github.com/zarr-developers/zarr-specs/issues/288)
            chunk_gt_data[chunk_i:] = -1
            if(chunk_i > 0):
                vcf_zarr.append(chunk_gt_data)
                vcf_zarr_meta.append(chunk_vardata)

                chunk_time = time.time()
                print(f"({overall_var_i}): Chunk time: {chunk_time - start_time}")

            vcf_f.close()

            end_time = time.time()
            print(f"Total time: {end_time - orig_start_time}")

    # If we're not combining, return the list of Zarr files
    if(not create_combined):
        return zarr_list
    
    # Write the combined Zarr reference file
    # This will be a JSON reference to the individual Zarr files
    combined_zarr_filepath = write_concatenated_zarr(zarr_list, combined_zarr_filepath, do_virtual=True)

    return combined_zarr_filepath




def vcfs_to_zarr(vcfs, output_folder, create_combined=True,
                               combined_prefix='combined_data', 
                               overwrite_individual=False,
                               overwrite_combined=False,
                               chunksize=2**10,
                               comm=None, rank=0, worldsize=1, 
                               parallel=False):
    """
    Converts a list of VCF files to a genotype matrix file in a specified format.
    Will make an output file for each VCF file and then either return a list of the files

    OR, if create_combined is True, will concatenate them all together 
    and then return the path to the concatenated file.
    """

    # TODO: make this blocking for each rank and only use rank 0 for making the folder
    os.makedirs(output_folder, exist_ok=True)

    if(isinstance(vcfs, str)):
        vcfs = [vcfs]


    if(worldsize == 1):
        print("Detected only one process, converting VCFs to genotype matrices in serial...")

        return vcfs_to_zarr_serial(vcfs, output_folder, create_combined=create_combined,
                                    combined_prefix=combined_prefix, 
                                    overwrite_individual=overwrite_individual, 
                                    overwrite_combined=overwrite_combined,
                                    chunksize=chunksize)

    else:
        return vcfs_to_zarr_parallel(vcfs, output_folder, 
                                        comm, rank, worldsize,
                                        create_combined=create_combined,
                                        combined_prefix=combined_prefix, 
                                        overwrite_individual=overwrite_individual, 
                                        overwrite_combined=overwrite_combined,
                                        chunksize=chunksize, 
                                        multivcf_parallel=parallel)
    


def _get_variant_properties(variant):
    """
    Extracts the variant properties from a VCF variant object
    as made in CyVCF2 and returns as a dictionary
    """
    return {
        'CHROM': variant.CHROM,
        'POS': variant.POS,
        'ID': variant.ID if variant.ID is not None else '.',
        'REF': variant.REF,
        'ALT': '/'.join(variant.ALT) if len(variant.ALT) > 0 else '',
        'AAF': variant.aaf,
        'CALL_RATE': variant.call_rate, 
        'N_CALLED': variant.num_called,
        'N_HOMREF': variant.num_hom_ref,
        'N_HOMALT': variant.num_hom_alt,
        'N_HET': variant.num_het,
        'N_UNKNOWN': variant.num_unknown,
        'FILTER': variant.FILTER if variant.FILTER is not None else '.',
        'N_PHASED': variant.gt_phases.sum(),
        'N_ALLELES': len(variant.ALT) + 1,
    }