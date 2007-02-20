// $COPYRIGHT$

#ifndef MTL_SHIFT_BLOCKS_DETAIL_INCLUDE
#define MTL_SHIFT_BLOCKS_DETAIL_INCLUDE

#include <algorithm>
#include <boost/numeric/mtl/mtl_exception.hpp>

namespace mtl { namespace operations { namespace detail {

template <typename Size, typename Starts, typename NewStarts, typename Ends, typename Data>
inline void copy_blocks_forward(Size& i, Size blocks, Starts const& starts, NewStarts const& new_starts, 
			 Ends const& ends, Data& data)
{
    using std::copy;

    // Copy forward as long as blocks are not shifted 
    for (; i < blocks && starts[i] >= new_starts[i]; ++i) 	
	if (starts[i] > new_starts[i])
	    copy(&data[starts[i]], &data[ends[i]], &data[new_starts[i]]);
}

template <typename Size, typename Starts, typename NewStarts, typename Ends, typename Data>
inline void copy_blocks_backward(Size& i, Size blocks, Starts const& starts, NewStarts const& new_starts, 
			  Ends const& ends, Data& data)
{
    using std::copy;
    using std::copy_backward;

    Size first = i;
    // find first block to be copied forward (or end)
    while (i < blocks && starts[i] < new_starts[i]) ++i;

    for (Size j = i; j-- > first; )
	if (ends[j] <= new_starts[j])
	    copy(&data[starts[j]], &data[ends[j]], &data[new_starts[j]]);
	else
	    copy_backward(&data[starts[j]], &data[ends[j]], &data[new_starts[j]+ends[j]-starts[j]]);
}

}}} // namespace mtl::operations::detail

#endif // MTL_SHIFT_BLOCKS_DETAIL_INCLUDE
