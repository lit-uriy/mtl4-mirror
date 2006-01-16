// $COPYRIGHT$

#ifndef MTL_DETAIL_RANGE_GENERATOR_INCLUDE
#define MTL_DETAIL_RANGE_GENERATOR_INCLUDE

#include <boost/numeric/mtl/glas_tags.hpp>
#include <boost/numeric/mtl/detail/base_cursor.hpp>

namespace mtl { namespace traits { namespace detail {

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

    // Cursor to some submatrix (e.g. row, column, block matrix, block row)
    // This cursor is intended to be used by range generators to iterate 
    // over subsets of the submatrix this cursor refers to.
    // For instance if this cursor refers to a row then a range 
    // can iterate over the elements in this row.
    // If this cursor refers to a block then a range can iterate over the rows in this block.
    // The level of a generated cursor must be of course at least one level less
    // The tag serves to dispatching between row and column cursors
    template <typename Collection, typename Tag, int Level = 2>
    struct sub_matrix_cursor
	: mtl::detail::base_cursor<int>
    {
	typedef mtl::detail::base_cursor<int>    base;
	static int const            level = Level;

	sub_matrix_cursor(int i, Collection const& c)
	    : base(i), ref(c) 
	{}	
	Collection const& ref;
    };

    template <typename Collection, typename Complexity, int Level = 2>
    struct all_rows_range_generator
    {
	typedef Complexity          complexity;
	static int const            level = Level;
	typedef sub_matrix_cursor<Collection, glas::tags::row_t, Level> type;

	type begin(Collection const& c)
	{
	    return type(c.begin_row(), c);
	}
	type end(Collection const& c)
	{
	    return type(c.end_row(), c);
	}
    };

    template <typename Collection, typename Complexity, int Level = 2>
    struct all_cols_range_generator
    {
	typedef Complexity          complexity;
	static int const            level = Level;
	typedef sub_matrix_cursor<Collection, glas::tags::col_t, Level> type;

	type begin(Collection const& c)
	{
	    return type(c.begin_col(), c);
	}
	type end(Collection const& c)
	{
	    return type(c.end_col(), c);
	}
    };

}}} // namespace mtl::traits::detail

#endif // MTL_DETAIL_RANGE_GENERATOR_INCLUDE
