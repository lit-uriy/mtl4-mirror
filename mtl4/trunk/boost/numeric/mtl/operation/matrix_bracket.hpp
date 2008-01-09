// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MATRIX_BRACKETS_INCLUDE
#define MTL_MATRIX_BRACKETS_INCLUDE

namespace mtl { namespace operations {

    template <typename Matrix, typename Ref, typename ValueRef>
    struct bracket_proxy
    {
	typedef typename Matrix::value_type   value_type;
	typedef typename Matrix::size_type    size_type;

	explicit bracket_proxy(Ref matrix, size_type row) : matrix(matrix), row(row) {}

	ValueRef operator[] (size_type col)
	{
	    return matrix(row, col);
	}

      protected:
	Ref         matrix;
	size_type   row;
    };
} // namespace operations

} // NAMESPACE mtl

#endif // MTL_MATRIX_BRACKETS_INCLUDE
