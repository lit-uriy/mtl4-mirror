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

#include <boost/shared_ptr.hpp>
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
    typedef mtl::matrix::mat_expr< self >                   expr_base;
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

    banded_view(boost::shared_ptr<Matrix> p, bsize_type begin, bsize_type end) 
	: base(p->get_dimensions()), my_copy(p), ref(*p), begin(begin), end(end) 
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
    //template <typename> friend struct ::mtl::sub_matrix_t<self>;

  protected:
    boost::shared_ptr<Matrix>           my_copy;
  public:
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
	    typedef mtl::matrix::banded_view<Matrix>                    view_type;
    	
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
	    view_type const&                                view;
	    typename mtl::traits::row<Matrix>::type         its_row;
	    typename mtl::traits::col<Matrix>::type         its_col;
	    typename mtl::traits::const_value<Matrix>::type its_value;
        };

    } // detail


}} // namespace mtl::matrix

namespace mtl { namespace traits {

    template <typename Matrix> 
    struct row<mtl::matrix::banded_view<Matrix> >
    {
	// from map_view
	typedef detail::mapped_row<sfunctor::identity<typename Matrix::value_type>, Matrix>   type;
    };

    template <typename Matrix> 
    struct col<mtl::matrix::banded_view<Matrix> >
    {
	// from map_view
	typedef detail::mapped_col<sfunctor::identity<typename Matrix::value_type>, Matrix>   type;
    };

    template <typename Matrix> 
    struct const_value<mtl::matrix::banded_view<Matrix> >
    {
	typedef mtl::matrix::detail::banded_value<Matrix>  type;
    };

    // ================
    // Range generators
    // ================

    // Use range_generator of original matrix
    template <typename Tag, typename Matrix> 
    struct range_generator<Tag, mtl::matrix::banded_view<Matrix> >
	: public detail::referred_range_generator<mtl::matrix::banded_view<Matrix>, 
						  range_generator<Tag, Matrix> >
    {};

    // To disambigue
    template <typename Matrix> 
    struct range_generator<tag::major, mtl::matrix::banded_view<Matrix> >
	: public detail::referred_range_generator<mtl::matrix::banded_view<Matrix>, 
						  range_generator<tag::major, Matrix> >
    {};


}} // mtl::traits


namespace mtl {

// ==========
// Sub matrix
// ==========

template <typename Matrix>
struct sub_matrix_t< mtl::matrix::banded_view<Matrix> >
{
    typedef mtl::matrix::banded_view<Matrix>                                           view_type;

    // Mapping of sub-matrix type
    typedef typename sub_matrix_t<Matrix>::sub_matrix_type                        ref_sub_type;
    typedef mtl::matrix::banded_view<ref_sub_type>                                     const_sub_matrix_type;
    typedef mtl::matrix::banded_view<ref_sub_type>                                     sub_matrix_type;
    typedef typename view_type::size_type                                         size_type;

    sub_matrix_type operator()(view_type const& view, size_type begin_r, size_type end_r, 
				     size_type begin_c, size_type end_c)
    {
	typedef boost::shared_ptr<ref_sub_type>                        pointer_type;

	// Submatrix of referred matrix (or view)
	// Create a submatrix, whos address will be kept by banded_view
	pointer_type p(new ref_sub_type(sub_matrix(view.ref, begin_r, end_r, begin_c, end_c)));
	return sub_matrix_type(p, view.begin, view.end); 
    }
};

} // mtl

#endif // MTL_MATRIX_BANDED_VIEW_INCLUDE
