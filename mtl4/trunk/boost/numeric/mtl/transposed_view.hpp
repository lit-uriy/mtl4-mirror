// $COPYRIGHT$

#ifndef MTL_TRANSPOSED_VIEW_INCLUDE
#define MTL_TRANSPOSED_VIEW_INCLUDE

#include <boost/numeric/mtl/traits.hpp>
#include <boost/numeric/mtl/detail/crtp_base_matrix.hpp>
#include <boost/numeric/mtl/operations/sub_matrix.hpp>

namespace mtl {

template <class Matrix> class transposed_view 
  : public detail::const_crtp_base_matrix< transposed_view<Matrix>, typename Matrix::value_type, typename Matrix::size_type >
{
    typedef transposed_view               self;
public:	
    typedef Matrix                        other;
    typedef typename transposed_orientation<typename Matrix::orientation>::type orientation;
    typedef typename Matrix::index_type                index_type;
    typedef typename Matrix::value_type                value_type;
#if 0 // not sure if I want these anymore
    typedef typename Matrix::pointer_type              pointer_type;
    typedef typename Matrix::const_pointer_type        const_pointer_type;
#endif
    typedef typename Matrix::key_type                  key_type;
    typedef typename Matrix::size_type                 size_type;
    typedef typename Matrix::dim_type::transposed_type dim_type;
    // typedef typename Matrix::el_cursor_type el_cursor_type;
    // typedef std::pair<el_cursor_type, el_cursor_type> el_cursor_pair;

    transposed_view (other& ref) : ref(ref) {}
    
//     el_cursor_pair elements() const 
//     {
//         return ref.elements();
//     }

    value_type operator() (std::size_t r, std::size_t c) const
    { 
        return ref(c, r); 
    }

    std::size_t dim1() const 
    { 
        return ref.dim2(); 
    }
    std::size_t dim2() const 
    { 
        return ref.dim1(); 
    }
    
    // Do we really need this?
    std::size_t offset(const value_type* p) const 
    { 
        return ref.offset(p); 
    }

#if 0 // seems dumb
    const_pointer_type elements() const 
    {
        return ref.elements(); 
    }
#endif

    dim_type dimensions() const 
    {
        return ref.dimensions().transpose(); 
    }

    std::size_t begin_row() const
    {
	return ref.begin_col();
    }

    std::size_t end_row() const
    {
	return ref.end_col();
    }

    std::size_t begin_col() const
    {
	return ref.begin_row();
    }

    std::size_t end_col() const
    {
	return ref.end_row();
    }


    other& ref;
};
  

namespace traits {

    template <class Matrix> struct is_mtl_type<transposed_view<Matrix> > 
    {
	static bool const value= is_mtl_type<Matrix>::value; 
    };

    template <class Matrix> 
    struct matrix_category<transposed_view<Matrix> >
    {
	typedef typename matrix_category<Matrix>::type type;
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

} // namespace traits


// ================
// Range generators
// ================

namespace traits
{

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
	          boost::is_same<typename range_generator<UseTag, Matrix>::complexity, complexity::infinite>
	        , range_generator<glas::tags::unsupported_t, Matrix>
	        , range_transposer_impl<UseTag, Matrix>
	      >::type {};
    }

    // Row and column cursors are interchanged
    template <class Matrix>
    struct range_generator<glas::tags::col_t, transposed_view<Matrix> >
	: detail::range_transposer<glas::tags::row_t, Matrix>
    {};

    template <class Matrix>
    struct range_generator<glas::tags::row_t, transposed_view<Matrix> >
	: detail::range_transposer<glas::tags::col_t, Matrix>
    {};

    // Other cursors are still use the same tag, e.g. elements
    template <class Tag, class Matrix>
    struct range_generator<Tag, transposed_view<Matrix> >
	: detail::range_transposer<Tag, Matrix>
    {};

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
    
    sub_matrix_type operator()(matrix_type& matrix, size_type begin_r, size_type end_r, size_type begin_c, size_type end_c)
    {
	// Submatrix of referred matrix, colums and rows interchanged
	typename sub_matrix_t<Matrix>::sub_matrix_type   sub(sub_matrix(matrix.ref, begin_c, end_c, begin_r, end_r));
	return sub_matrix_type(sub);
    }
    
    const_sub_matrix_type operator()(matrix_type const& matrix, size_type begin_r, size_type end_r, 
				     size_type begin_c, size_type end_c)
    {
	// Submatrix of referred matrix, colums and rows interchanged
	typename sub_matrix_t<Matrix>::const_sub_matrix_type   sub(sub_matrix(matrix.ref, begin_c, end_c, begin_r, end_r));
	return const_sub_matrix_type(sub);
    }
};

} // namespace mtl

#endif // MTL_TRANSPOSED_VIEW_INCLUDE



