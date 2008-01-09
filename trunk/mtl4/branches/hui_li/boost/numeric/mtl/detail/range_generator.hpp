// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_DETAIL_RANGE_GENERATOR_INCLUDE
#define MTL_DETAIL_RANGE_GENERATOR_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/glas_tag.hpp>
#include <boost/numeric/mtl/detail/base_cursor.hpp>

namespace mtl { namespace traits { namespace detail {

    /// Range generator that traverses all elements of some densely stored collection 
    /** - Or contiguous parts of such collection
        - Works for matrices and vectors when derived from contiguous_memory_block
    **/
    template <typename Collection, typename Cursor, typename Complexity>
    struct dense_element_range_generator
    {
	typedef Complexity          complexity;
	typedef Cursor              type;
	static int const            level = 1;

	type begin(Collection const& collection)
	{
	    return collection.elements();
	}
	type end(Collection const& collection)
	{
	    return collection.elements() + collection.used_memory();
	}
    };


    // Like above over all elements but in terms of offsets
    // Also with reference to collection in cursor
    template <typename Matrix, typename Cursor, typename Complexity>
    struct all_offsets_range_generator
    {
	typedef Complexity          complexity;
	typedef Cursor              type;
	static int const            level = 1;

	type begin(Matrix const& matrix) const
	{
	    return type(matrix, 0);
	}
	type end(Matrix const& matrix) const
	{
	    return type(matrix, matrix.nnz());
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
    template <typename Matrix, typename Tag, int Level = 2>
    struct sub_matrix_cursor
	: mtl::detail::base_cursor<int>
    {
	typedef sub_matrix_cursor                self;
	typedef mtl::detail::base_cursor<int>    base;
	static int const            level = Level;

	sub_matrix_cursor(int i, Matrix const& c)
	    : base(i), ref(c) 
	{}	

	self operator+(int offset) const
	{
	    return self(key + offset, ref);
	    // return base::operator+(offset);
	}
	
	Matrix const& ref;
    };


    template <typename Matrix, typename Complexity, int Level = 2>
    struct all_rows_range_generator
    {
	typedef Complexity          complexity;
	static int const            level = Level;
	typedef sub_matrix_cursor<Matrix, glas::tag::row, Level> type;

	type begin(Matrix const& c)
	{
	    return type(c.begin_row(), c);
	}
	type end(Matrix const& c)
	{
	    return type(c.end_row(), c);
	}
    };


    template <typename Matrix, typename Complexity, int Level = 2>
    struct all_cols_range_generator
    {
	typedef Complexity          complexity;
	static int const            level = Level;
	typedef sub_matrix_cursor<Matrix, glas::tag::col, Level> type;

	type begin(Matrix const& c)
	{
	    return type(c.begin_col(), c);
	}
	type end(Matrix const& c)
	{
	    return type(c.end_col(), c);
	}
    };

    // Use RangeGenerator for Collection by applying to .ref
    template <typename Collection, typename RangeGenerator>
    struct referred_range_generator
    {
	typedef typename RangeGenerator::complexity complexity;
	static int const                            level = RangeGenerator::level;
	typedef typename RangeGenerator::type       type;
	
	type begin(const Collection& c)
	{
	    return RangeGenerator().begin(c.ref);
	}

	type end(const Collection& c)
	{
	    return RangeGenerator().end(c.ref);
	}
    };

}}} // namespace mtl::traits::detail

#endif // MTL_DETAIL_RANGE_GENERATOR_INCLUDE
