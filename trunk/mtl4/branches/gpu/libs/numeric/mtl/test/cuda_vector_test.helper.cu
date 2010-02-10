
#include <boost/numeric/mtl/cuda/vector_kernel.cu>

namespace mtl { namespace cuda {

	void dummy () {
	    vec_rscale_asgn<int> v1;
	    dim3 dimGrid(1), dimBlock(5); 
	    launch_function<<<dimGrid, dimBlock>>>(v1);
	}
}}
