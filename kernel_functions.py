import cupy as cp



# Function to initialize all kernels (so that they aren't recompiled on every iteration)
def initialize_kernels(userdef_threshold_dict, n_samples):
    """
    Initializes all kernels for the LD calculation
    """
    # Initialize all kernels
    kernels = {}

    for metric, thresh_val in userdef_threshold_dict.items():
        kernels[f"threshold_{metric}_kernel"] = create_inplace_threshold_kernel(threshold=thresh_val,
                                                                            ret_val=0, comparison_str='<', 
                                                                            kernel_name=f"threshold_{metric}_kernel")
    
    kernels['dmax_kernel'] = create_dmax_kernel()
    kernels['rsq_from_dsq_kernel'] = create_rsq_from_dsq_kernel()
    kernels['dprime_lowsamp_kernel'] = create_inplace_threshold_kernel(threshold=0.8*n_samples*2, ret_val='nan', comparison_str='<', kernel_name='dprime_lowsamp_kernel')
    
    return kernels


# Elementwise threshold kernel to reduce memory usage
def create_inplace_threshold_kernel(threshold, ret_val, comparison_str='<', kernel_name='threshold_kernel'):
    """
    Creates a threshold kernel for various uses
    Allows one to determine a threshold, a return value, and a comparison
    Will return a kernel that will set values in-place based on the comparison
    That kernel will compare x to the threshold using `comparison_str` and 
    set it to `ret_val` if the comparison is true
    note, this works for >, < (not tested otherwise, but >= and <= should work too)
    """
    if(str(ret_val).lower() == 'nan'):
        ret_val = '0.0 / 0.0'

    return cp.ElementwiseKernel(
        'T x',
        'T y',
        f'y = x {comparison_str} {threshold} ? {ret_val} : x',
        kernel_name,
        no_return=True
    )


# Kernel to compute Dmax from D, pA, and pB
# TODO: extend this to the entire D' calculation? 
# Probably not, since this would require overloading 
# SGEMM which is extremely well-optimized and not worth doing
def create_dmax_kernel():
    """
    Creates a kernel to compute Dmax from D, pA, and pB
    by using the cuLaunchKernel interface with an ElementwiseKernel
    """
    input_vars = 'float32 D, raw float32 pA, raw float32 pB, int32 n_cols'
    output_vars = 'float32 out'

    operation = r'''
int row = i / n_cols;
int col = i % n_cols;

if (D < 0) {
    out = min(pA[row] * pB[col], (1-pA[row]) * (1-pB[col]));
} else {
    out = min(pA[row] * (1-pB[col]), (1-pA[row]) * pB[col]);
}

'''
# float max_p = max(max(pA[row],1-pA[row]),max(pB[col],1-pB[col]));
# if (max_p > 0.999) {
#     out = 0;
#     return;
# }


    return cp.ElementwiseKernel(
        input_vars,
        output_vars,
        operation,
    )


# Kernel to compute Rsq from Dsq, pA, and pB
def create_rsq_from_dsq_kernel():
    """
    Creates a kernel to compute Rsq from Dsq, pA, and pB
    by using the cuLaunchKernel interface with an ElementwiseKernel
    """
    input_vars = 'float32 Dsq, raw float32 pA, raw float32 pB, int32 n_cols'
    output_vars = 'float32 out'

    operation = r'''
int row = i / n_cols;
int col = i % n_cols;

out = Dsq / (pA[row] * (1-pA[row]) * pB[col] * (1-pB[col]));

'''
    return cp.ElementwiseKernel(
        input_vars,
        output_vars,
        operation,
    )