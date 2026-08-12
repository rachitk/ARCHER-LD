import zarr
import numcodecs
import numpy as np
from kerchunk.zarr import single_zarr
from kerchunk.combine import concatenate_arrays, merge_vars
import ujson


def write_concatenated_zarr(zarr_list, out_file, do_virtual=False, do_pad=True):
    """
    Uses Kerchunk to either:
    
    concatenate many Zarr files in a 'virtual' dataset approach
    where the combined file is really a reference to existing files on disk
    (do_virtual is True in this case and the output array will always be padded
    so we will pad the attributes with -1 or '' as needed)
    
    OR

    concatenate many Zarr files and write the combined data to a new Zarr file
    where the combined file is itself a Zarr file with no references
    (do_pad if requested is only used in this case to ensure the final size is 
    a multiple of the chunksize)
    """
    # Concatenate the Zarr files
    zarr_shapes = []
    zarr_f_handles = []
    kerchunk_zarr_json = []

    for zarr_filename in zarr_list:
        if(zarr_filename.endswith('.json')):
            zarr_f = zarr.open_array('reference://', storage_options={'fo': zarr_filename}, mode='r')
        else:
            zarr_f = zarr.open(zarr_filename, mode='r')
        
        zarr_f_handles.append(zarr_f)

        zarr_geno_shape = zarr_f['gt'].shape
        zarr_shapes.append(zarr_geno_shape)
        
        kerchunk_zarr_json.append(single_zarr(zarr_filename))

    sub_arr_names = list(zarr_f.array_keys())

    nonvar_axis_shapes = [dset_shape[1:] for dset_shape in zarr_shapes]

    if not (nonvar_axis_shapes.count(nonvar_axis_shapes[0]) == len(nonvar_axis_shapes)):
        raise ValueError("Number of samples (or other non-variant axis) in each Zarr file is not the same. Cannot concatenate.")
    
    if(do_virtual):
        # Use Kerchunk to concatenate the Zarr files
        # this makes a digital array that is a reference to the individual Zarr files
        out_json_path = out_file
        sub_arr_combined = []

        for arr_name in sub_arr_names:
            combined_arr_json = concatenate_arrays(kerchunk_zarr_json, axis=0, path=arr_name, check_arrays=True)
            sub_arr_combined.append(combined_arr_json)

        combined_zarr_json = merge_vars(sub_arr_combined)

        # Save this to a file
        with open(out_json_path, "wb") as combined_zarr_f:
            combined_zarr_f.write(ujson.dumps(combined_zarr_json).encode())

        # Reload file as a Zarr reference store array to create attributes as needed
        # TODO: add any attributes that we would like here
        # combined_zarr = zarr.open('reference://', storage_options={'fo': out_json_path}, mode='r+')

        # Can load this file back in with:
        # combined_zarr = zarr.open('reference://', storage_options={'fo': out_json_path}, mode='r')

        return out_json_path

    else:
        # Open a Zarr group, then loop over the arrays in each Zarr file and concatenate them
        out_zarr_path = out_file
        combined_zarr = zarr.open(out_zarr_path, mode='w', zarr_format=2)

        for arr_name in sub_arr_names:
            # Create a Zarr array within the group to hold the concatenated data
            # (this will be slower than the virtual approach and take more space)

            chunksize = zarr_f_handles[0][arr_name].chunks
            dtype = zarr_f_handles[0][arr_name].dtype
            arr_shape = zarr_f_handles[0][arr_name].shape

            n_rows = sum([dset_shape[0] for dset_shape in zarr_shapes])
            
            if(do_pad):
                # Ensure the final size is a multiple of the chunksize
                # as otherwise this can cause issues down the line with Zarr
                finalchunk_offset = chunksize[0] - (n_rows % chunksize[0])
                if finalchunk_offset == chunksize[0]:
                    finalchunk_offset = 0
                total_rows = n_rows + finalchunk_offset
            else:
                finalchunk_offset = 0
                total_rows = n_rows

            subarr_arr = combined_zarr.create(arr_name, shape=(total_rows,) + arr_shape[1:],
                                    chunks=chunksize, dtype=dtype,
                                    config={'write_empty_chunks': False})
        
            start_i = 0
            for zarr_f in zarr_f_handles:
                zarr_f_arr = zarr_f[arr_name]
                end_i = start_i + zarr_f_arr.shape[0]
                subarr_arr[start_i:end_i] = zarr_f_arr[:]
                start_i = end_i

            # For all data, we need to replace with -1 (for numeric) or '' (for string) if we need padding
            def dtype_to_pad(dtype_in):
                if np.issubdtype(dtype_in, np.number):
                    return -1
                elif (dtype_in.kind == 'U') or (dtype_in.kind == 'O'):
                    return ''
                elif dtype_in.names is not None:
                    return tuple(dtype_to_pad(dtype_i[0]) for dtype_i in dtype_in.fields.values())
                else:
                    raise ValueError(f"Unsupported dtype {dtype_in} for padding.")



            # Replace the rest of the array with -1 or '' as needed
            if do_pad and finalchunk_offset > 0:
                expand_dim_nums = list(range(1, len(arr_shape)))
                pad_vals = [dtype_to_pad(dtype)] * (finalchunk_offset)
                pad_vals = np.array(pad_vals, dtype=dtype)

                pad_vals = np.broadcast_to(np.expand_dims(pad_vals,expand_dim_nums), subarr_arr[-finalchunk_offset:].shape)

                subarr_arr[-finalchunk_offset:] = pad_vals


        # Can load this file back in with:
        # combined_zarr = zarr.open(out_zarr_path, mode='r')
        
        return out_zarr_path
    

def _preinit_property_dict(num_vars):
    """
    Preinitializes a dictionary of variant properties
    for a given number of variants
    Note, we assume that REF and ALT strings are less than 30 characters,
    that the chromosome string is less than 10 characters,
    that the ID is less than 30 characters,
    and that there are less than 2 billion of:
        bases in any chromosome,
        called genotypes,
        and number of samples

    CHANGE: Change all string types to object types to allow for variable length strings
    If we support numpy 2.x only, we can use the new string type that supports this instead
    (Note that object-type arrays are much slower and more costly on disk than fixed-length arrays)
    Original lengths:
    - CHROM: 10
    - ID: 30
    - REF: 30
    - ALT: 30
    - FILTER: 30
    """
    
    # TODO: Check if StringDType is available (numpy 2.0+) and supported in structured arrays
    # if so, this is a better dtype to use for strings

    # All string types previously set to 'object', but this is no longer supported in Zarr-Python 3
    # so we have converted them to fixed-length string types of a larger length but this will still absolutely cause truncation 
    # TODO: change this to variable-length string types when Zarr finally gets back around to supporting this in a reasonable way (if ever?) or when Numpy 2.x supports StringDType in structured arrays

    return {
        'CHROM': np.full(num_vars, '', dtype=np.dtype('U8')),
        'POS': np.full(num_vars, -1, dtype=np.int32), 
        'ID': np.full(num_vars, '', dtype=np.dtype('U128')),
        'REF': np.full(num_vars, '', dtype=np.dtype('U128')),
        'ALT': np.full(num_vars, '', dtype=np.dtype('U32')),
        'AAF': np.full(num_vars, -1., dtype=np.float32),
        'CALL_RATE': np.full(num_vars, -1., dtype=np.float32),
        'N_CALLED': np.full(num_vars, -1, dtype=np.int32),
        'N_HOMREF': np.full(num_vars, -1, dtype=np.int32),
        'N_HOMALT': np.full(num_vars, -1, dtype=np.int32),
        'N_HET': np.full(num_vars, -1, dtype=np.int32),
        'N_UNKNOWN': np.full(num_vars, -1, dtype=np.int32),
        'FILTER': np.full(num_vars, '', dtype=np.dtype('U128')),
        'N_PHASED': np.full(num_vars, -1, dtype=np.int32),
        'N_ALLELES': np.full(num_vars, -1, dtype=np.int8), #if there are more than 128 alt alleles, the genotype matrix is also going to break (is also an int8)
    }


def _property_dict_to_structured_array(prop_dict):
    """
    Converts a dictionary of variant properties to a structured numpy array
    where each key in the dictionary is a field in the array, and dtypes are
    inferred from the data types of the numpy arrays in the dictionary
    """
    # NOTE: object dtypes no longer supported in Zarr-Python 3, so we need to convert these to fixed-length string types of a larger length (U50) which should allow for variable length strings up to 50 characters 
    dtypes = [v.dtype if v.dtype != object else np.dtype('U50') for v in prop_dict.values()]

    s_array = np.array(list(zip(*prop_dict.values())), dtype=list(zip(prop_dict.keys(), dtypes)))
    
    return s_array