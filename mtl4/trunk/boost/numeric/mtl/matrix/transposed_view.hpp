// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_TRANSPOSED_VIEW_INCLUDE
#define MTL_TRANSPOSED_VIEW_INCLUDE

#include <boost/shared_ptr.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/matrix/crtp_base_matrix.hpp>
#include <boost/numeric/mtl/operation/sub_matrix.hpp>
#include <boost/numeric/mtl/matrix/mat_expr.hpp>

namespace mtl { namespace matrix {


// Orientation type for transposed matrix
template <class T> struct transposed_orientation {};

template<> struct transposed_orientation<tag::row_major> 
{
    typedef tag::col_major type; 
};

template<> struct transposed_orientation<tag::col_major> 
{
    typedef tag::row_major type; 
};



template <class Matrix> 
struct transposed_view 
  : public boost::mpl::if_<
          boost::is_const<Matrix>
        , const_crtp_base_matrix< const transposed_view<const Matrix>, 
				  typename Matrix::value_type, typename Matrix::size_type >
        , crtp_base_matrix< transposed_view<Matrix>, 
			    typename Matrix::value_type, typename Matrix::size_type >
      >::type,
    public matrix::mat_expr< transposed_view<Matrix> >
{
    typedef transposed_view               self;
    typedef mat_expr< self >              expr_base;
    typedef Matrix                        other;

    typedef typename transposed_orientation<typename Matrix::orientation>::type orientation;
    typedef typename Matrix::index_type                index_type;
    typedef typename Matrix::value_type                value_type;
    typedef typename Matrix::const_reference           const_reference;
    typedef typename Matrix::key_type                  key_type;
    typedef typename Matrix::size_type                 size_type;
    typedef typename Matrix::dim_type::transposed_type dim_type;
    typedef matrix::parameters<orientation, index_type, dim_type> parameters;

    typedef typename boost::mpl::if_<boost::is_const<Matrix>,
				     const_reference,
				     value_type&
				    >::type                  access_type;

    typedef typename boost::mpl::if_<boost::is_const<Matrix>,
				     const Matrix&,
				     Matrix&
				    >::type                  ref_type;


    transposed_view (ref_type ref) : /* expr_base(*this), */ ref(ref) {}
    
    transposed_view (boost::shared_ptr<Matrix> p) : /* expr_base(*this), */ my_copy(p), ref(*p) {}
    
    const_reference operator() (size_type r, size_type c) const
    { 
        return ref(c, r); 
    }

    access_type operator() (size_type r, size_type c)
    { 
        return ref(c, r); 
    }

    size_type dim1() const 
    { 
        return ref.dim2(); 
    }
    size_type dim2() const 
    { 
        return ref.dim1(); 
    }
    
    dim_type dimensions() const 
    {
        return ref.dimensions().transpose(); 
    }

    size_type begin_row() const
    {
	return ref.begin_col();
    }

    size_type end_row() const
    {
	return ref.end_col();
    }

    size_type num_rows() const
    {
	return ref.num_cols();
    }

    size_type begin_col() const
    {
	return ref.begin_row();
    }

    size_type end_col() const
    {
	return ref.end_row();
    }

    size_type num_cols() const
    {
	return ref.num_rows();
    }

    size_type nnz() const
    {
	return ref.nnz();
    }
    
protected:
    boost::shared_ptr<Matrix>           my_copy;
public:
    ref_type                            ref;
};
  

// ================
// Free functions
// ================

template <typename Matrix>
typename transposed_view<Matrix>::size_type
inline num_rows(const transposed_view<Matrix>& A)
{
    return A.num_rows();
}

template <typename Matrix>
typename transposed_view<Matrix>::size_type
inline num_cols(const transposed_view<Matrix>& A)
{
    return A.num_cols();
}

template <typename Matrix>
typename transposed_view<Matrix>::size_type
inline size(const transposed_view<Matrix>& A)
{
    return A.num_cols() * A.num_rows();
}

// ==========
// Sub matrix
// ==========

template <typename Matrix>
struct sub_matrix_t< transposed_view<Matrix> >
{
    typedef transposed_view<Matrix>                                               matrix_type;

    // Transposed of submatrix type
    typedef transposed_view<typename sub_matrix_t<Matrix>::sub_matrix_type>       sub_matrix_type;
    typedef transposed_view<typename sub_matrix_t<Matrix>::const_sub_matrix_type> const_sub_matrix_type;
    typedef typename matrix_type::size_type                                       size_type;
    
    sub_matrix_type operator()(matrix_type& A, size_type begin_r, size_type end_r, size_type begin_c, size_type end_c)
    {
	typedef typename sub_matrix_t<Matrix>::sub_matrix_type   ref_sub_type;
	typedef boost::shared_ptr<ref_sub_type>                  pointer_type;

	// Submatrix of referred matrix, colums and rows interchanged
	// Create a submatrix, whos address will be kept by transposed_view
	pointer_type p(new ref_sub_type(sub_matrix(A.ref, begin_c, end_c, begin_r, end_r)));
	return sub_matrix_type(p); 
    }
    
    const_sub_matrix_type operator()(matrix_type const& A, size_type begin_r, size_type end_r, 
				     size_type begin_c, size_type end_c)
    {
	typedef typename sub_matrix_t<Matrix>::const_sub_matrix_type   ref_sub_type;
	typedef boost::shared_ptr<ref_sub_type>                        pointer_type;

	// Submatrix of referred matrix, colums and rows interchanged
	// Create a submatrix, whos address will be kept by transposed_view
	pointer_type p(new ref_sub_type(sub_matrix(A.ref, begin_c, end_c, begin_r, end_r)));
	return const_sub_matrix_type(p); 
    }

};

}} // mtl::matrix

namespace mtl { namespace traits {

    template <class Matrix> 
    struct category<transposed_view<Matrix> >
    {
	typedef typename category<Matrix>::type type;
    };

    namespace detail {

	template <class Matrix> 
	struct transposed_row
	{
	    typedef typename Matrix::key_type   key_type;
	    typedef typename Matrix::size_type  size_type;
    	
	    transposed_row(transposed_view<Matrix> const& transposed_matrix) 
		: its_col(transposed_matrix.ref) {}

	    size_type operator() (key_type const& key) const
	    {
		return its_col(key);
	    }

	  protected:
	    typename col<Matrix>::type  its_col;
        };


        template <class Matrix> 
        struct transposed_col
        {
	    typedef typename Matrix::key_type   key_type;
	    typedef typename Matrix::size_type  size_type;
    	
	    transposed_col(transposed_view<Matrix> const& transposed_matrix) 
		: its_row(transposed_matrix.ref) {}

	    size_type operator() (key_type const& key) const
	    {
		return its_row(key);
	    }

          protected:
	    typename row<Matrix>::type  its_row;
        };
	
    } // namespace detail
        
    template <class Matrix> 
    struct row<transposed_view<Matrix> >
    {
	typedef detail::transposed_row<Matrix>  type;
    };

    template <class Matrix> 
    struct col<transposed_view<Matrix> >
    {
	typedef detail::transposed_col<Matrix>  type;
    };

    template <class Matrix> 
    struct const_value<transposed_view<Matrix> >
    {
	typedef mtl::detail::const_value_from_other<transposed_view<Matrix> > type;
    };

    template <class Matrix> 
    struct value<transposed_view<Matrix> >
    {
	typedef mtl::detail::value_from_other<transposed_view<Matrix> > type;
    };


// ================
// Range generators
// ================

    namespace detail
    {
	template <class UseTag, class Matrix>
	struct range_transposer_impl
	{
	    typedef range_generator<UseTag, Matrix>  generator;
	    typedef typename generator::complexity   complexity;
	    typedef typename generator::type         type;
	    static int const                         level = generator::level;
	    type begin(transposed_view<Matrix> const& m)
	    {
		return generator().begin(m.ref);
	    }
	    type end(transposed_view<Matrix> const& m)
	    {
		return generator().end(m.ref);
	    }
	};

	// If considered range_generator for Matrix isn't supported, i.e. has infinite complexity
	// then define as unsupported for transposed view 
	// (range_transposer_impl wouldn't compile in this case)
	template <class UseTag, class Matrix>
	struct range_transposer
	    : boost::mpl::if_<
	          boost::is_same<typename range_generator<UseTag, Matrix>::complexity, complexity_classes::infinite>
	        , range_generator<tag::unsupported, Matrix>
	        , range_transposer_impl<UseTag, Matrix>
	      >::type {};
    }

    // Row and column cursors are interchanged
    template <class Matrix>
    struct range_generator<glas::tag::col, transposed_view<Matrix> >
	: detail::range_transposer<glas::tag::row, Matrix>
    {};

    template <class Matrix>
    struct range_generator<glas::tag::row, transposed_view<Matrix> >
	: detail::range_transposer<glas::tag::col, Matrix>
    {};

    // To traverse the major dimension refer to the Matrix
    template <class Matrix>
    struct range_generator<tag::major, transposed_view<Matrix> >
	: detail::range_transposer<tag::major, Matrix>
    {};

    // Other cursors are still use the same tag, e.g. elements
    template <class Tag, class Matrix>
    struct range_generator<Tag, transposed_view<Matrix> >
	: detail::range_transposer<Tag, Matrix>
    {};


}} // namespace mtl::traits

#endif // MTL_TRANSPOSED_VIEW_INCLUDE



