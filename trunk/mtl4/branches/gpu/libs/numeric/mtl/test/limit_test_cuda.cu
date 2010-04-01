#include <iostream>
#include <cassert>


#if 0
#include <boost/static_assert.hpp> // ok
// #include <boost/type_traits.hpp> // nicht ok
#include <boost/type_traits/remove_const.hpp> // ok
#include <boost/lambda/lambda.hpp> // ok
#include <boost/utility/enable_if.hpp>
#include <boost/numeric/ublas/detail/returntype_deduction.hpp>
#include <boost/mpl/at.hpp>


#include "boost/type_traits/add_const.hpp"
#include "boost/type_traits/add_cv.hpp"
#include "boost/type_traits/add_pointer.hpp"
#include "boost/type_traits/add_reference.hpp"
#include "boost/type_traits/add_volatile.hpp"
//#include "boost/type_traits/alignment_of.hpp"
#include "boost/type_traits/has_nothrow_assign.hpp"
#include "boost/type_traits/has_nothrow_constructor.hpp"
#include "boost/type_traits/has_nothrow_copy.hpp"
#include "boost/type_traits/has_nothrow_destructor.hpp"
#include "boost/type_traits/has_trivial_assign.hpp"
#include "boost/type_traits/has_trivial_constructor.hpp"
#include "boost/type_traits/has_trivial_copy.hpp"
#include "boost/type_traits/has_trivial_destructor.hpp"
#include "boost/type_traits/has_virtual_destructor.hpp"
#include "boost/type_traits/is_signed.hpp"
#include "boost/type_traits/is_unsigned.hpp"
#include "boost/type_traits/is_abstract.hpp"
#include "boost/type_traits/is_arithmetic.hpp"
#include "boost/type_traits/is_array.hpp"
#include "boost/type_traits/is_base_and_derived.hpp"
#include "boost/type_traits/is_base_of.hpp"
#include "boost/type_traits/is_class.hpp"
#include "boost/type_traits/is_compound.hpp"
#include "boost/type_traits/is_const.hpp"
#include "boost/type_traits/is_convertible.hpp"
#include "boost/type_traits/is_empty.hpp"
#include "boost/type_traits/is_enum.hpp"
#include "boost/type_traits/is_float.hpp"
#include "boost/type_traits/is_floating_point.hpp"
#include "boost/type_traits/is_function.hpp"
#include "boost/type_traits/is_fundamental.hpp"
#include "boost/type_traits/is_integral.hpp"
#include "boost/type_traits/is_member_function_pointer.hpp"
#include "boost/type_traits/is_member_object_pointer.hpp"
#include "boost/type_traits/is_member_pointer.hpp"
#include "boost/type_traits/is_object.hpp"
#include "boost/type_traits/is_pod.hpp"
#include "boost/type_traits/is_polymorphic.hpp"
#include "boost/type_traits/is_pointer.hpp"
#include "boost/type_traits/is_reference.hpp"
#include "boost/type_traits/is_same.hpp"
#include "boost/type_traits/is_scalar.hpp"
#include "boost/type_traits/is_stateless.hpp"
#include "boost/type_traits/is_union.hpp"
#include "boost/type_traits/is_void.hpp"
#include "boost/type_traits/is_volatile.hpp"
//#include "boost/type_traits/rank.hpp"
//#include "boost/type_traits/extent.hpp"
#include "boost/type_traits/remove_bounds.hpp"
#include "boost/type_traits/remove_extent.hpp"
#include "boost/type_traits/remove_all_extents.hpp"
#include "boost/type_traits/remove_const.hpp"
#include "boost/type_traits/remove_cv.hpp"
#include "boost/type_traits/remove_pointer.hpp"
#include "boost/type_traits/remove_reference.hpp"
#include "boost/type_traits/remove_volatile.hpp"
//#include "boost/type_traits/type_with_alignment.hpp"
#include "boost/type_traits/function_traits.hpp"
//#include "boost/type_traits/aligned_storage.hpp"
#include "boost/type_traits/floating_point_promotion.hpp"
#include "boost/type_traits/integral_promotion.hpp"
#include "boost/type_traits/promote.hpp"
#include <boost/type_traits/make_unsigned.hpp>
#include <boost/type_traits/make_signed.hpp>
#include <boost/type_traits/decay.hpp>
#include <boost/type_traits/is_complex.hpp>
#include "boost/type_traits/ice.hpp"

#endif

#include <boost/mpl/if.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/int.hpp>

#define BLOCK_SIZE 512
__global__ void vector_plus(int *d_out, int *d_in, int dim)
{
    unsigned int block_id=  blockIdx.x + gridDim.x * blockIdx.y;
    unsigned int thread_id= blockDim.x * block_id + threadIdx.x;
    if(thread_id < dim)
        d_out[thread_id] += d_in[thread_id];
}

////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
    // pointer for host memory and size
    int *h_a;
    int dimA = 500*1000*1000;

    int *d_b, *d_a;

    size_t memSize = dimA * sizeof(int);
    h_a = (int *) malloc(memSize);

    cudaMalloc( (void **) &d_a, memSize );

    cudaMalloc( (void **) &d_b, memSize );

    for (int i = 0; i < dimA; i++)
        h_a[i] = 2;
    cudaMemcpy( d_a, h_a, memSize, cudaMemcpyHostToDevice );
    for (int i = 0; i < dimA; i++)
        h_a[i] = 1;
    cudaMemcpy( d_b, h_a, memSize, cudaMemcpyHostToDevice );
    dim3 dimBlock(BLOCK_SIZE);
    int num_blocks = (int) ((float) (dimA + BLOCK_SIZE - 1) / (float) BLOCK_SIZE);
    int max_blocks_per_dimension = 65535;
    int num_blocks_y = (int) ((float) (num_blocks + max_blocks_per_dimension - 1) / (float) max_blocks_per_dimension);
    int num_blocks_x = (int) ((float) (num_blocks + num_blocks_y - 1) / (float) num_blocks_y);
    dim3 grid_size(num_blocks_x, num_blocks_y, 1);
    std::cout<< "dimA/65535/65535= "<< double((dimA/65535)/double(65535)) << "\n";
    std::cout<< "dimA/65535= "<< dimA/65535 << "\n";
    std::cout<< "num_blocks= "<< num_blocks << "\n";
    std::cout<< "num_blocks_x= "<< num_blocks_x << "\n";
    std::cout<< "num_blocks_y= "<< num_blocks_y << "\n";

    vector_plus<<< grid_size, dimBlock >>>( d_b, d_a , dimA);
    cudaThreadSynchronize();
    cudaMemcpy( h_a, d_b, memSize, cudaMemcpyDeviceToHost );
    for (int i = 0; i < 20; i++)
      assert(h_a[i] == 3 );
      std::cout<< "h_a[" << dimA-1 << "]=" << h_a[dimA-1] << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    return 0;
}

