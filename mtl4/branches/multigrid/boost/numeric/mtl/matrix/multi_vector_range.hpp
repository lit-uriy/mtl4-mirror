// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MATRIX_MULTI_VECTOR_RANGE_INCLUDE
#define MTL_MATRIX_MULTI_VECTOR_RANGE_INCLUDE

#include <boost/numeric/mtl/matrix/multi_vector.hpp>

namespace mtl { namespace matrix {



template <typename Vector>
class multi_vector_range
{
    typedef multi_vector<Vector>                     ref_type;
    typedef multi_vector_range                       self;
    typedef typename Collection<Vector>::size_type   size_type;

  public:
    
    multi_vector_range(ref_type& ref, irange const& r) 
      : ref(ref), range(intersection(r, irange(0, num_cols(ref)))) {} {}
    

    /// Number of columns
    friend size_type num_cols(const self& A) { return range.size(); }
    /// Number of rows
    friend size_type num_rows(const self& A) { return num_rows(ref); }

    const_reference operator() (size_type i, size_type j) const { return data[j][i+range.start()]; }
    reference operator() (size_type i, size_type j) { return data[j][i+range.start()]; }

    Vector& vector(size_type i) { return data[i+range.start()]; }
    const Vector& vector(size_type i) const { return data[i+range.start()]; }

  private:
    ref_type& ref;
    irange    range;
};



}} // namespace mtl::matrix

#endif // MTL_MATRIX_MULTI_VECTOR_RANGE_INCLUDE
