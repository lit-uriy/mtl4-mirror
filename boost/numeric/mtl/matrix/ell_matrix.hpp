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

#ifndef MTL_MATRIX_ELL_MATRIX_INCLUDE
#define MTL_MATRIX_ELL_MATRIX_INCLUDE

#include <vector>
#include <cassert>

#include <boost/static_assert.hpp>
#include <boost/numeric/mtl/matrix/parameter.hpp>
#include <boost/numeric/mtl/matrix/compressed2D.hpp>
#include <boost/numeric/mtl/utility/wrapped_object.hpp>
#include <boost/numeric/mtl/utility/is_row_major.hpp>
#include <boost/numeric/mtl/operation/std_output_operator.hpp>


namespace mtl { namespace matrix {

/// Matrix in Ell-Pack format; still in early stage, to be used with care (if at all)
template <typename Value, typename Parameters = matrix::parameters<> >
class ell_matrix
  : public base_matrix<Value, Parameters>,
    public const_crtp_base_matrix< ell_matrix<Value, Parameters>, Value, typename Parameters::size_type >,
    public crtp_matrix_assign< ell_matrix<Value, Parameters>, Value, typename Parameters::size_type >,
    public mat_expr< ell_matrix<Value, Parameters> >
{
    BOOST_STATIC_ASSERT((mtl::traits::is_row_major<Parameters>::value));

    typedef base_matrix<Value, Parameters>             super;
    typedef ell_matrix                                 self;
    typedef mat_expr< ell_matrix<Value, Parameters> >  expr_base;

  public:
    typedef Parameters                                 parameters;
    typedef typename Parameters::orientation           orientation;
    typedef typename Parameters::dimensions            dimensions;
    typedef Value                                      value_type;
    typedef value_type                                 const_reference;

    typedef typename Parameters::size_type             size_type;
    typedef crtp_matrix_assign<self, Value, size_type> assign_base;

    static const unsigned alignment=                   32; // TBD: make more flexible later

    /// Construct matrix of size \p num_rows times \p num_cols
    explicit ell_matrix (size_type num_rows, size_type num_cols)
      : super(non_fixed::dimensions(num_rows, num_cols)), 
	stride((this->dim1() + alignment - 1) / alignment * alignment), slots(0), inserting(false)
    {}

    /// Print internal representation
    template <typename OStream>
    void print_internal(OStream& os) const
    {
#     ifdef MTL_HAS_STD_OUTPUT_OPERATOR
	os << "indices = " << indices << '\n'; 
	os << "values  = " << data << '\n';
#     endif
    }

    /// Entry in row \p r and column \p c
    value_type operator()(size_type r, size_type c) const
    {
	for (size_type k= r, i= 0; i < slots; ++i, k+= stride)
	    if (indices[k] == c)
		return data[k];
	return value_type(0);
    }

    template <typename VectorIn, typename VectorOut, typename Assign>
    void mult_rmajor_vector(const VectorIn& v, VectorOut& w, Assign) const
    {
	for (size_type r= 0; r < this->dim1(); ++r) {
	    value_type s(0);
	    for (size_type k= r, i= 0; i < slots; ++i, k+= stride)
		s+= data[k] * v[indices[k]];
	    Assign::first_update(w[r], s);
	}
    }

  protected:
    void allocate_slots(size_type s)
    {
	slots= s;
	size_type size= stride * s;
	indices.resize(size); data.resize(size);
    }

    template <typename V, typename P, typename Updater> friend struct ell_matrix_inserter; 

    std::vector<value_type> data; 
    std::vector<size_type>  indices;
    size_type               stride, slots;
    bool                    inserting;
};


template <typename Value, typename Parameters, typename Updater = mtl::operations::update_store<Value> >
struct ell_matrix_inserter
  : wrapped_object<compressed2D<Value, Parameters> >,
    compressed2D_inserter<Value, Parameters, Updater>
{
    typedef typename Parameters::size_type    size_type;
    typedef Value                             value_type;
    typedef ell_matrix<Value, Parameters>     matrix_type;
    typedef compressed2D<Value, Parameters>   compressed_type;
    typedef wrapped_object<compressed_type>   wrapped_type;
    typedef compressed2D_inserter<Value, Parameters, Updater>   base_inserter;
    
    explicit ell_matrix_inserter(matrix_type& A, size_type slot_size = 5)
      : wrapped_type(num_rows(A), num_cols(A)),
	base_inserter(wrapped_type::wrapped_object_member, slot_size),
	A(A)
    {
	A.inserting= true;
    }

    ~ell_matrix_inserter()
    {
	this->finish();
	const compressed_type& B= this->wrapped_object_member;
	// std::cout << "Finished insertion!\nA (compressed2D) is:\n" << B;
	
	size_type max_slots= 0;
	for (size_type i= 0; i < B.dim1(); ++i) {
	    size_type s= this->slot_ends[i] - this->starts[i];	
	    if (s > max_slots)
		max_slots= s;
	}
	A.allocate_slots(max_slots);

	for (size_type i= 0; i < B.dim1(); ++i) {
	    size_type patch_entries= max_slots - (this->slot_ends[i] - this->starts[i]), k= i,
		      patch_index= 0;
	    for (size_type j= this->starts[i]; j < this->slot_ends[i]; ++j, k+= A.stride) {
		patch_index= A.indices[k]= B.ref_minor()[j];
		A.data[k]= B.data[j];
	    }
	    for (size_type j= 0; j < patch_entries; ++j, k+= A.stride) {
		A.indices[k]= patch_index;
		A.data[k]= value_type(0);
	    }
	}

	A.inserting= false;
    }

    matrix_type& A;
};

// ================
// Free functions
// ================

template <typename Value, typename Parameters>
typename ell_matrix<Value, Parameters>::size_type
inline num_rows(const ell_matrix<Value, Parameters>& matrix)
{
    return matrix.num_rows();
}

template <typename Value, typename Parameters>
typename ell_matrix<Value, Parameters>::size_type
inline num_cols(const ell_matrix<Value, Parameters>& matrix)
{
    return matrix.num_cols();
}

template <typename Value, typename Parameters>
// typename ell_matrix<Value, Parameters>::size_type risks overflow
std::size_t
inline size(const ell_matrix<Value, Parameters>& matrix)
{
    return std::size_t(matrix.num_cols()) * std::size_t(matrix.num_rows());
}



}} // namespace mtl::matrix

#endif // MTL_MATRIX_ELL_MATRIX_INCLUDE
