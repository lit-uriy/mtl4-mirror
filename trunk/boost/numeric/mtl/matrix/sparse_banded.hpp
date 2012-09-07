// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG, www.simunova.com. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also tools/license/license.mtl.txt in the distribution.

#ifndef MTL_MATRIX_SPARSE_BANDED_INCLUDE
#define MTL_MATRIX_SPARSE_BANDED_INCLUDE

#include <vector>

#include <boost/numeric/mtl/matrix/dimension.hpp>
#include <boost/numeric/mtl/matrix/parameter.hpp>
#include <boost/numeric/mtl/matrix/base_matrix.hpp>
#include <boost/numeric/mtl/matrix/crtp_base_matrix.hpp>
#include <boost/numeric/mtl/matrix/mat_expr.hpp>

namespace mtl { namespace matrix {

/// Sparse banded matrix class
template <typename Value, typename Parameters = matrix::parameters<> >
class sparse_banded
  : public base_matrix<Value, Parameters>,
    public mat_expr< sparse_banded<Value, Parameters> >
{
    typedef std::size_t                                size_t;
    typedef base_matrix<Value, Parameters>             super;
    typedef sparse_banded<Value, Parameters>           self;

  public:
    typedef Value                                      value_type;
    typedef typename Parameters::size_type             size_type;

    /// Construct matrix of dimension \p nr by \p nc
    sparse_banded(size_type nr, size_type nc) 
      : super(non_fixed::dimensions(nr, nc)), data(0), inserting(false)
    {}

    ~sparse_banded() { delete[] data; }

  private:
    std::vector<size_t>       bands;
    value_type*               data;
    bool                      inserting;
};

/// Inserter for sparse banded matrix
template <typename Value, typename Parameters, typename Updater = mtl::operations::update_store<Value> >
struct sparse_banded_inserter
{
  private:
    typedef sparse_banded<Value, Parameters>                                 matrix_type;
    typedef typename Parameters::size_type                                   size_type;

  public:
    /// Construct inserter for matrix \p A; second argument for slot_size ignored
    sparse_banded_inserter(matrix_type& A, size_type) : A(A) 
    {
	A.inserting= true;
    }

  private:
    matrix_type&       A;
};

}} // namespace mtl::matrix

#endif // MTL_MATRIX_SPARSE_BANDED_INCLUDE
