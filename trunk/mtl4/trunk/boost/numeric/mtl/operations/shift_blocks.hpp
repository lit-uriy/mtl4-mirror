// $COPYRIGHT$

#ifndef MTL_SHIFT_BLOCKS_INCLUDE
#define MTL_SHIFT_BLOCKS_INCLUDE

#include <boost/numeric/mtl/operations/shift_blocks_detail.hpp>

namespace mtl { namespace operations {

// Shift blocks in an 1D array to remove unnecessary holes
// inserting holes in other places where needed (e.g. for inserting new values)
//
// Block 'i' is the half-open interval [starts[i], ends[i]) in data
// It will be copied into [new_starts[i], ...) in place
// Blocks are ordered: start[i] <= start[i+1]
// Data between blocks are considered holes and can be overwritten
//
template <typename Size, typename Starts, typename NewStarts, typename Ends, typename Data>
void shift_blocks(Size blocks, Starts const& starts, NewStarts const& new_starts, 
		  Ends const& ends, Data& data)
{
    for (Size i = 0; i < blocks; ) {
	detail::copy_blocks_forward(i, blocks, starts, new_starts, ends, data);
	detail::copy_blocks_backward(i, blocks, starts, new_starts, ends, data);
    }
}


}} // namespace mtl::operations

#endif // MTL_SHIFT_BLOCKS_INCLUDE
