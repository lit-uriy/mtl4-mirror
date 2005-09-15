// $COPYRIGHT$

#ifndef MTL_DETAIL_RANGE_GENERATOR_INCLUDE
#define MTL_DETAIL_RANGE_GENERATOR_INCLUDE

namespace mtl { namespace detail {

    // Range generator that traverses all elements of some densely stored collection 
    // or contiguous parts of such collection
    // works if Collection is derived from base_matrix
    template <typename Collection, typename Cursor, typename Complexity>
    struct dense_element_range_generator
    {
	typedef Complexity          complexity;
	static int const            level = 1;
	typedef Cursor              type;
	type begin(Collection const& c)
	{
	    return c.elements();
	}
	type end(Collection const& c)
	{
	    return c.elements() + c.num_elements();
	}
    };

 }} // namespace mtl::detail

#endif // MTL_DETAIL_RANGE_GENERATOR_INCLUDE
