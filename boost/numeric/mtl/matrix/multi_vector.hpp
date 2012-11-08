// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MATRIX_MULTI_VECTOR_INCLUDE
#define MTL_MATRIX_MULTI_VECTOR_INCLUDE

#include <boost/utility/enable_if.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/matrix/parameter.hpp>
#include <boost/numeric/mtl/matrix/crtp_base_matrix.hpp>
#include <boost/numeric/mtl/matrix/mat_expr.hpp>
#include <boost/numeric/mtl/matrix/multi_vector_range.hpp>
#include <boost/numeric/mtl/utility/is_what.hpp>
#include <boost/numeric/mtl/utility/is_multi_vector_expr.hpp>
#include <boost/numeric/mtl/vector/parameter.hpp>



// Under development (to be used with caution)

namespace mtl { namespace matrix {

    
// Might need to be defined later
struct multi_vector_key {};

/// Matrix constituting of set of column vectors (under development)
template <typename Vector>
class multi_vector
  : public base_matrix<typename mtl::Collection<Vector>::value_type, parameters<> >,
    public crtp_base_matrix< multi_vector<Vector>, typename Collection<Vector>::value_type, 
			     typename Collection<Vector>::size_type>,
    public mat_expr< multi_vector<Vector> >    
{
    typedef base_matrix<typename Collection<Vector>::value_type, parameters<> >           super;

    // Vector must by column vector
    BOOST_STATIC_ASSERT((boost::is_same<typename OrientedCollection<Vector>::orientation,
			                tag::col_major>::value));
  public:
    typedef multi_vector                             self;
    // typedef mtl::matrix::parameters<>                parameters;
    typedef Vector                                   vector_type;
    typedef tag::col_major                           orientation;
    typedef typename Collection<Vector>::value_type  value_type;
    typedef typename Collection<Vector>::size_type   size_type;
    typedef const value_type&                        const_reference;
    typedef value_type&                              reference;
    typedef multi_vector_key                         key_type;
    typedef crtp_matrix_assign< self, value_type, size_type >    assign_base;

    multi_vector() : super(non_fixed::dimensions(0, 0)) {}

    /// Constructor by number of rows and columns
    multi_vector(size_type num_rows, size_type num_cols)
      : super(non_fixed::dimensions(num_rows, num_cols)), 
	data(num_cols, Vector(num_rows))
    {
	this->my_nnz= num_rows * num_cols;
    }

    /// Constructor column vector and number of columns (for easier initialization)
    multi_vector(const Vector& v, size_type num_cols)
      : super(non_fixed::dimensions(size(v), num_cols)),
	data(num_cols, v)
    {
	this->my_nnz= num_cols * size(v);
    }

    /// Change dimension, can keep old data
    void change_dim(size_type r, size_type c)
    {
	super::change_dim(r, c);
	data.change_dim(c);
	for (size_type i= 0; i < c; i++)
	    data[i].change_dim(r);
    }

    // Todo: multi_vector with other matrix expressions
    /// Assign multi_vector and expressions thereof, general matrices currently not allowed 
    template <typename Src>
    typename boost::enable_if<mtl::traits::is_multi_vector_expr<Src>, self&>::type
    operator=(const Src& src)
    {
//        MTL_THROW_IF(num_rows(src) != super::num_rows() || num_cols(src) != super::num_cols(), incompatible_size());
	MTL_THROW_IF((mtl::matrix::num_rows(src) != super::num_rows() || mtl::matrix::num_cols(src) != super::num_cols()), incompatible_size());
	for (std::size_t i= 0, n= super::num_cols(); i < n; ++i)
	    vector(i)= src.vector(i);
	return *this;
    }

    template <typename Src>
    typename boost::enable_if_c<mtl::traits::is_matrix<Src>::value 
				&& !mtl::traits::is_multi_vector_expr<Src>::value, self&>::type
    operator=(const Src& src)
    {
	assign_base::operator=(src);
	return *this;
    }
    
    /// Assign scalar
    template <typename Src>
    typename boost::enable_if<mtl::traits::is_scalar<Src>, self&>::type
    operator=(const Src& src)
    {
	assign_base::operator=(src);
	return *this;
    }

    const_reference operator() (size_type i, size_type j) const { return data[j][i]; }
    reference operator() (size_type i, size_type j) { return data[j][i]; }

    Vector& vector(size_type i) { return data[i]; }
    const Vector& vector(size_type i) const { return data[i]; }

    Vector& at(size_type i) { return data[i]; }
    const Vector& at(size_type i) const { return data[i]; }

    multi_vector_range<Vector> vector(irange const& r) const { return multi_vector_range<Vector>(*this, r); }

  protected:  
    mtl::vector::dense_vector<Vector, mtl::vector::parameters<> >          data;
};

/// Number of rows
template< typename Vector >
typename Collection< Vector >::size_type num_cols(const multi_vector< Vector >& A) { return A.num_cols(); }

/// Number of columns
template< typename Vector >
typename Collection< Vector >::size_type num_rows(const multi_vector< Vector >& A) { return A.num_rows(); }

/// Size as defined by number of rows times columns
template< typename Vector >
typename Collection< Vector >::size_type size(const multi_vector< Vector >& A) { return num_rows(A) * num_cols(A); }
}} // namespace mtl::matrix

namespace mtl {
	using matrix::multi_vector;
}

#endif // MTL_MATRIX_MULTI_VECTOR_INCLUDE
