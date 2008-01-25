// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_HERMITIAN_VIEW_INCLUDE
#define MTL_HERMITIAN_VIEW_INCLUDE

#include <boost/numeric/mtl/matrix/map_view.hpp>
#include <boost/numeric/mtl/matrix/transposed_view.hpp>
#include <boost/numeric/mtl/operation/conj.hpp>


namespace mtl { namespace matrix {

template <class Matrix> 
struct hermitian_view 
    : public map_view<sfunctor::conj<typename Matrix::value_type>, 
		      transposed_view<Matrix> >
{
    typedef sfunctor::conj<typename Matrix::value_type>            functor_type;
    typedef map_view<functor_type, transposed_view<Matrix> >       base;

    hermitian_view(const Matrix& matrix)
	: trans_view(const_cast<Matrix&>(matrix)), base(functor_type(), trans_view)
    {}
    
#if 0
    hermitian_view(boost::shared_ptr<Matrix> p)
	: trans_view(p), base(functor_type(), trans_view)
    {}
#endif

  private:
    transposed_view<Matrix>  trans_view;
};


}} // namespace mtl::matrix

#endif // MTL_HERMITIAN_VIEW_INCLUDE
