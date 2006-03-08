// $COPYRIGHT$

#ifndef MTL_DENSE2D_INCLUDE
#define MTL_DENSE2D_INCLUDE

#include <algorithm>
#include <boost/numeric/mtl/detail/base_cursor.hpp>
#include <boost/numeric/mtl/detail/strided_base_cursor.hpp>
#include <boost/numeric/mtl/detail/base_matrix.hpp>
#include <boost/numeric/mtl/detail/contiguous_memory_matrix.hpp>
#include <boost/numeric/mtl/detail/range_generator.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/property_map.hpp>
#include <boost/numeric/mtl/range_generator.hpp>
#include <boost/numeric/mtl/detail/range_generator.hpp>
#include <boost/numeric/mtl/complexity.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>

namespace mtl {

using std::size_t;

// Forward declarations
template <typename Elt, typename Parameters> class dense2D;
struct dense2D_indexer;


// Cursor over every element
template <class Elt> 
struct dense_el_cursor : public detail::base_cursor<const Elt*> 
{
    typedef Elt                           value_type;
    typedef const value_type*             const_pointer_type; // ?
    typedef detail::base_cursor<const Elt*> super;

    dense_el_cursor () {} 
    dense_el_cursor (const_pointer_type me) : super(me) {}

    template <typename Parameters>
    dense_el_cursor(dense2D<Elt, Parameters> const& ma, size_t r, size_t c)
	: super(ma.elements() + ma.indexer(ma, r, c))
    {}
};

// Cursor over strided elements
template <class Elt> 
struct strided_dense_el_cursor : public detail::strided_base_cursor<const Elt*> 
{
    typedef Elt                           value_type;
    typedef const value_type*             const_pointer_type; // ?
    typedef detail::strided_base_cursor<const Elt*> super;

    strided_dense_el_cursor () {} 
    strided_dense_el_cursor (const_pointer_type me, size_t stride) : super(me, stride) {}

    template <typename Parameters>
    strided_dense_el_cursor(dense2D<Elt, Parameters> const& ma, size_t r, size_t c, size_t stride)
	: super(ma.elements() + ma.indexer(ma, r, c), stride)
    {}
};

// Indexing for dense matrices
struct dense2D_indexer 
{
private:
    // helpers for public functions
    size_t offset(size_t dim2, size_t r, size_t c, row_major) const 
    {
	return r * dim2 + c; 
    }
    size_t offset(size_t dim2, size_t r, size_t c, col_major) const 
    {
	return c * dim2 + r; 
    }
    
    size_t row(size_t offset, size_t dim2, row_major) const 
    {
	return offset / dim2; 
    }
    size_t row(size_t offset, size_t dim2, col_major) const 
    {
	return offset % dim2;
    }
    
    size_t col(size_t offset, size_t dim2, row_major) const 
    {
	return offset % dim2;
    }
    size_t col(size_t offset, size_t dim2, col_major) const 
    {
	return offset / dim2; 
    }

 public:
    template <class Matrix>
    size_t operator() (const Matrix& ma, size_t r, size_t c) const
    {
	// convert into c indices
	typename Matrix::index_type my_index;
	size_t my_r= index::change_from(my_index, r);
	size_t my_c= index::change_from(my_index, c);
	return offset(ma.dim2(), my_r, my_c, typename Matrix::orientation());
    }

    template <class Matrix>
    size_t row(const Matrix& ma, typename Matrix::key_type key) const
    {
	// row with c-index for my orientation
	size_t r= row(ma.offset(key), ma.dim2(), typename Matrix::orientation());
	return index::change_to(typename Matrix::index_type(), r);
    }

    template <class Matrix>
    size_t col(const Matrix& ma, typename Matrix::key_type key) const 
    {
	// column with c-index for my orientation
	size_t c= col(ma.offset(key), ma.dim2(), typename Matrix::orientation());
	return index::change_to(typename Matrix::index_type(), c);
    }
}; // dense2D_indexer


namespace detail 
{
    
    // Compute required memory
    // Enabling mechanism to make sure that computation is valid
    template <typename Parameters, bool Enable>
    struct dense2D_array_size {
	static std::size_t const value= 0;
    };

    template <typename Parameters>
    struct dense2D_array_size<Parameters, true>
    {
	typedef typename Parameters::dimensions   dimensions;
	BOOST_STATIC_ASSERT((dimensions::is_static));
	static std::size_t const value= dimensions::Num_Rows * dimensions::Num_Cols;
    };

} // namespace detail

  
// Dense 2D matrix type
template <typename Elt, typename Parameters>
class dense2D : public detail::base_matrix<Elt, Parameters>, 
		public detail::contiguous_memory_matrix< Elt, Parameters::on_stack, 
							 detail::dense2D_array_size<Parameters, Parameters::on_stack>::value >
{
    typedef dense2D                                    self;
    typedef detail::base_matrix<Elt, Parameters>       super;
  public:
    typedef Parameters                        parameters;
    typedef typename Parameters::orientation  orientation;
    typedef typename Parameters::index        index_type;
    typedef typename Parameters::dimensions   dim_type;
    typedef Elt                               value_type;

    typedef detail::contiguous_memory_matrix<Elt, Parameters::on_stack, 
					     detail::dense2D_array_size<Parameters, Parameters::on_stack>::value>     super_memory;

    typedef const value_type*                 const_pointer_type;
    typedef const_pointer_type                key_type;
    typedef std::size_t                       size_type;
    typedef dense_el_cursor<Elt>              el_cursor_type;  
    typedef dense2D_indexer                   indexer_type;

  protected:
    void set_nnz()
    {
      this->nnz = this->dim.num_rows() * this->dim.num_cols();
    }

  public:
    // if compile time matrix size allocate memory
    dense2D() : super(), super_memory(dim_type().num_rows() * dim_type().num_cols()) 
    {
	set_nnz();
    }

    // only sets dimensions, only for run-time dimensions
    explicit dense2D(mtl::non_fixed::dimensions d) 
	: super(d), super_memory(d.num_rows() * d.num_cols()) 
    {
	set_nnz();
    }

    // sets dimensions and pointer to external data
    explicit dense2D(mtl::non_fixed::dimensions d, value_type* a) 
      : super(d), super_memory(a) 
    { 
        set_nnz();
    }

    // same constructor for compile time matrix size
    // sets dimensions and pointer to external data
    explicit dense2D(value_type* a) : super(), super_memory(a) 
    { 
	BOOST_STATIC_ASSERT((dim_type::is_static));
        set_nnz();
    }

    // friend class indexer_type; should work without friend declaration

    // old style, better use value property map
    value_type operator() (size_t r, size_t c) const 
    {
      size_t offset= indexer(*this, r, c);
      return this->data[offset];
    }

    value_type& reference(size_t r, size_t c)
    {
	return this->data[indexer(*this, r, c)]; 
    }

    value_type const& reference(size_t r, size_t c) const
    {
	return this->data[indexer(*this, r, c)]; 
    }

    indexer_type  indexer;
 }; // dense2D



// =============
// Property Maps
// =============

namespace traits 
{
    template <class Elt, class Parameters>
    struct row<dense2D<Elt, Parameters> >
    {
        typedef mtl::detail::indexer_row_ref<dense2D<Elt, Parameters> > type;
    };

    template <class Elt, class Parameters>
    struct col<dense2D<Elt, Parameters> >
    {
        typedef mtl::detail::indexer_col_ref<dense2D<Elt, Parameters> > type;
    };

    template <class Elt, class Parameters>
    struct const_value<dense2D<Elt, Parameters> >
    {
        typedef mtl::detail::direct_const_value<dense2D<Elt, Parameters> > type;
    };

    template <class Elt, class Parameters>
    struct value<dense2D<Elt, Parameters> >
    {
        typedef mtl::detail::direct_value<dense2D<Elt, Parameters> > type;
    };

    template <class Elt, class Parameters>
    struct is_mtl_type<dense2D<Elt, Parameters> > 
    {
	static bool const value= true; 
    };

    // define corresponding type without all template parameters
    template <class Elt, class Parameters>
    struct matrix_category<dense2D<Elt, Parameters> > 
    {
	typedef tag::dense2D type;
    };

} // namespace traits
    
template <class Elt, class Parameters>
inline typename traits::row<dense2D<Elt, Parameters> >::type
row(dense2D<Elt, Parameters>  const& matrix)
{
    return typename traits::row<dense2D<Elt, Parameters> >::type(matrix);
}

template <class Elt, class Parameters>
inline typename traits::col<dense2D<Elt, Parameters> >::type
col(dense2D<Elt, Parameters>  const& matrix)
{
    return typename traits::col<dense2D<Elt, Parameters> >::type(matrix);
}

template <class Elt, class Parameters>
inline typename traits::const_value<dense2D<Elt, Parameters> >::type
const_value(dense2D<Elt, Parameters>  const& matrix)
{
    return typename traits::const_value<dense2D<Elt, Parameters> >::type(matrix);
}

template <class Elt, class Parameters>
inline typename traits::value<dense2D<Elt, Parameters> >::type
value(dense2D<Elt, Parameters> & matrix)
{
    return typename traits::value<dense2D<Elt, Parameters> >::type(matrix);
}



// ================
// Range generators
// ================

namespace traits
{
    template <class Elt, class Parameters>
    struct range_generator<glas::tags::all_t, dense2D<Elt, Parameters> >
      : detail::dense_element_range_generator<dense2D<Elt, Parameters>,
					      dense_el_cursor<Elt>, complexity::linear_cached>
    {};

    template <class Elt, class Parameters>
    struct range_generator<glas::tags::nz_t, dense2D<Elt, Parameters> >
      : detail::dense_element_range_generator<dense2D<Elt, Parameters>,
					      dense_el_cursor<Elt>, complexity::linear_cached>
    {};

    namespace detail 
    {
	// complexity of dense row cursor depends on storage scheme
	// if orientation is row_major then complexity is cached_linear, otherwise linear
	template <typename Orientation> struct dense2D_rc {};
	template<> struct dense2D_rc<row_major>
	{
	    typedef complexity::linear_cached type;
	};
	template<> struct dense2D_rc<col_major>
	{
	    typedef complexity::linear type;
	};

	// Complexity of column cursor is of course opposite
	template <typename Orientation> struct dense2D_cc
	    : dense2D_rc<typename transposed_orientation<Orientation>::type>
	{};
    }

    template <class Elt, class Parameters>
    struct range_generator<glas::tags::row_t, dense2D<Elt, Parameters> >
	: detail::all_rows_range_generator<dense2D<Elt, Parameters>, 
					   typename detail::dense2D_rc<typename Parameters::orientation>::type>
    {};
 
    // For a cursor pointing to some row give the range of elements in this row 
    template <class Elt, class Parameters>
    struct range_generator<glas::tags::nz_t, 
			   detail::sub_matrix_cursor<dense2D<Elt, Parameters>, glas::tags::row_t, 2> >
    {
	typedef dense2D<Elt, Parameters>  matrix;
	typedef detail::sub_matrix_cursor<matrix, glas::tags::row_t, 2> cursor;
	// linear for col_major and linear_cached for row_major
	typedef typename detail::dense2D_rc<typename Parameters::orientation>::type   complexity;
	static int const             level = 1;
	// for row_major dense_el_cursor would be enough, i.e. bit less overhead but uglier code
	typedef strided_dense_el_cursor<Elt> type;
	size_t stride(cursor const&, row_major)
	{
	    return 1;
	}
	size_t stride(cursor const& c, col_major)
	{
	    return c.ref.dim2();
	}
	type begin(cursor const& c)
	{
	    return type(c.ref, c.key, c.ref.begin_col(), stride(c, typename matrix::orientation()));
	}
	type end(cursor const& c)
	{
	    return type(c.ref, c.key, c.ref.end_col(), stride(c, typename matrix::orientation()));
	}
    };

    template <class Elt, class Parameters>
    struct range_generator<glas::tags::all_t, 
			   detail::sub_matrix_cursor<dense2D<Elt, Parameters>, glas::tags::row_t, 2> >
        : range_generator<glas::tags::nz_t, 
			  detail::sub_matrix_cursor<dense2D<Elt, Parameters>, glas::tags::row_t, 2> >
    {};



    template <class Elt, class Parameters>
    struct range_generator<glas::tags::col_t, dense2D<Elt, Parameters> >
	: detail::all_cols_range_generator<dense2D<Elt, Parameters>, 
					   typename detail::dense2D_cc<typename Parameters::orientation>::type>
    {};
 
    // For a cursor pointing to some row give the range of elements in this row 
    template <class Elt, class Parameters>
    struct range_generator<glas::tags::nz_t, 
			   detail::sub_matrix_cursor<dense2D<Elt, Parameters>, glas::tags::col_t, 2> >
    {
	typedef dense2D<Elt, Parameters>  matrix;
	typedef detail::sub_matrix_cursor<matrix, glas::tags::col_t, 2> cursor;	
	typedef typename detail::dense2D_cc<typename Parameters::orientation>::type   complexity;
	static int const             level = 1;
	typedef strided_dense_el_cursor<Elt> type;
	size_t stride(cursor const&, col_major)
	{
	    return 1;
	}
	size_t stride(cursor const& c, row_major)
	{
	    return c.ref.dim2();
	}
	type begin(cursor const& c)
	{
	    return type(c.ref, c.ref.begin_row(), c.key, stride(c, typename matrix::orientation()));
	}
	type end(cursor const& c)
	{
	    return type(c.ref, c.ref.end_row(), c.key, stride(c, typename matrix::orientation()));
	}
    };

    template <class Elt, class Parameters>
    struct range_generator<glas::tags::all_t, 
			   detail::sub_matrix_cursor<dense2D<Elt, Parameters>, glas::tags::col_t, 2> >
        : range_generator<glas::tags::nz_t, 
			  detail::sub_matrix_cursor<dense2D<Elt, Parameters>, glas::tags::col_t, 2> >
    {};



} // namespace traits


} // namespace mtl

#endif // MTL_DENSE2D_INCLUDE




