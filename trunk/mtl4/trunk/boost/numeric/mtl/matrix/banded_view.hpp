// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MATRIX_BANDED_VIEW_INCLUDE
#define MTL_MATRIX_BANDED_VIEW_INCLUDE

#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>
#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/detail/crtp_base_matrix.hpp>
#include <boost/numeric/mtl/detail/base_matrix.hpp>
#include <boost/numeric/mtl/operation/sfunctor.hpp>
#include <boost/numeric/mtl/matrix/mat_expr.hpp>
#include <boost/numeric/mtl/matrix/map_view.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>


// Is not mutable because masking out some values forbids returning references
//
// Arbitrary combinations with other views (using shared_ptr) is planned

namespace mtl { namespace matrix {

// Forward
namespace detail { 
    template <typename> struct banded_value; 
    template <typename, typename> struct mapped_value; 
}


template <typename Matrix> 
struct banded_view 
  : public mtl::detail::const_crtp_base_matrix< banded_view<Matrix>, 
						typename Matrix::value_type, typename Matrix::size_type >,
    public mat_expr< banded_view<Matrix> >,
    public mtl::detail::base_matrix<typename Matrix::value_type, typename Matrix::parameters>
{
    typedef banded_view                                self;
    typedef matrix::mat_expr< self >                   expr_base;
    typedef mtl::detail::base_matrix<typename Matrix::value_type, typename Matrix::parameters> base;
    
    typedef Matrix                                     other;
    typedef typename Matrix::orientation               orientation;
    typedef typename Matrix::index_type                index_type;
    typedef typename Matrix::parameters                parameters;

    typedef typename Matrix::value_type                value_type;
    typedef typename Matrix::const_reference           const_reference;

    typedef typename Matrix::key_type                  key_type;
    typedef typename Matrix::size_type                 size_type;
    typedef typename Matrix::dim_type                  dim_type;

    typedef long int                                   bsize_type;

    banded_view(const other& ref, bsize_type begin, bsize_type end) 
	: base(ref.get_dimensions()), ref(ref), begin(begin), end(end) 
    {}

    value_type operator() (size_type r, size_type c) const
    {
	using math::zero;
	bsize_type bc= static_cast<bsize_type>(c), br= static_cast<bsize_type>(r),
	           band= bc - br;
	// Need value to return correct zero as well (i.e. matrices itself)
	value_type v= ref(r, c); 
	return begin <= band && band < end ? v : zero(v);
    }

    // need const functions
    bsize_type get_begin() const { return begin; }
    bsize_type get_end() const { return end; }

    template <typename> friend struct detail::banded_value;
    template <typename, typename> friend struct detail::map_value;

    //protected:
    const other&      ref;
    bsize_type        begin, end;
};


// ================
// Free functions
// ================

template <typename Matrix>
typename banded_view<Matrix>::size_type
inline num_rows(const banded_view<Matrix>& matrix)
{
    return matrix.num_rows();
}

template <typename Matrix>
typename banded_view<Matrix>::size_type
inline num_cols(const banded_view<Matrix>& matrix)
{
    return matrix.num_cols();
}

template <typename Matrix>
typename banded_view<Matrix>::size_type
inline size(const banded_view<Matrix>& matrix)
{
    return matrix.num_cols() * matrix.num_rows();
}


    namespace detail {

	template <typename Matrix> 
	struct banded_value
	{
	    typedef typename Matrix::key_type                      key_type;
	    typedef typename Matrix::value_type                    value_type;
	    typedef matrix::banded_view<Matrix>                    view_type;
    	
	    banded_value(view_type const& view) 
		: view(view), its_row(view.ref), its_col(view.ref), its_value(view.ref) 
	    {}

	    value_type operator() (key_type const& key)
	    {
		using math::zero;
		typedef typename view_type::bsize_type   bsize_type;

		bsize_type br= static_cast<bsize_type>(its_row(key)), 
                           bc= static_cast<bsize_type>(its_col(key)),
		           band= bc - br;
		// Need value to return correct zero as well (i.e. matrices itself)
		const value_type v= its_value(key);

		return view.get_begin() <= band && band < view.get_end() ? v : zero(v);
	    }

	  protected:
	    view_type const&                           view;
	    typename traits::row<Matrix>::type         its_row;
	    typename traits::col<Matrix>::type         its_col;
	    typename traits::const_value<Matrix>::type its_value;
        };

    } // detail


}} // namespace mtl::matrix

namespace mtl { namespace traits {

    template <typename Matrix> 
    struct row<matrix::banded_view<Matrix> >
    {
	// from map_view
	typedef detail::mapped_row<sfunctor::identity<typename Matrix::value_type>, Matrix>   type;
    };

    template <typename Matrix> 
    struct col<matrix::banded_view<Matrix> >
    {
	// from map_view
	typedef detail::mapped_col<sfunctor::identity<typename Matrix::value_type>, Matrix>   type;
    };

    template <typename Matrix> 
    struct const_value<matrix::banded_view<Matrix> >
    {
	typedef matrix::detail::banded_value<Matrix>  type;
    };

    // ================
    // Range generators
    // ================

    // Use range_generator of original matrix
    template <typename Tag, typename Matrix> 
    struct range_generator<Tag, matrix::banded_view<Matrix> >
	: public detail::referred_range_generator<matrix::banded_view<Matrix>, 
						  range_generator<Tag, Matrix> >
    {};

    // To disambigue
    template <typename Matrix> 
    struct range_generator<tag::major, matrix::banded_view<Matrix> >
	: public detail::referred_range_generator<matrix::banded_view<Matrix>, 
						  range_generator<tag::major, Matrix> >
    {};


}} // mtl::traits



#endif // MTL_MATRIX_BANDED_VIEW_INCLUDE
