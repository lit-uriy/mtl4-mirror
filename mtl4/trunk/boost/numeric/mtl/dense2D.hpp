// $COPYRIGHT$

#ifndef MTL_DENSE2D_INCLUDE
#define MTL_DENSE2D_INCLUDE

#include <algorithm>
#include <boost/numeric/mtl/common_includes.hpp>
#include <boost/numeric/mtl/detail/base_cursor.hpp>
#include <boost/numeric/mtl/detail/base_sub_matrix.hpp>
#include <boost/numeric/mtl/detail/strided_base_cursor.hpp>
#include <boost/numeric/mtl/detail/contiguous_memory_matrix.hpp>


namespace mtl {

using std::size_t;

// Forward declarations
template <typename Value, typename Parameters> class dense2D;
struct dense2D_indexer;


// Cursor over every element
template <typename Value> 
struct dense_el_cursor : public detail::base_cursor<const Value*> 
{
    typedef Value                           value_type;
    typedef const value_type*             const_pointer_type; // ?
    typedef detail::base_cursor<const Value*> super;

    typedef dense_el_cursor               self;

    dense_el_cursor () {} 
    dense_el_cursor (const_pointer_type me) : super(me) {}

    template <typename Parameters>
    dense_el_cursor(dense2D<Value, Parameters> const& ma, size_t r, size_t c)
	: super(ma.elements() + ma.indexer(ma, r, c))
    {}

    self operator+(int x) const
    {
	return super::operator+(x);
    }

    self operator-(self const& x)
    {
	return super::operator-(x);
    }
};

// Cursor over strided elements
template <typename Value> 
struct strided_dense_el_cursor : public detail::strided_base_cursor<const Value*> 
{
    typedef Value                           value_type;
    typedef const value_type*             const_pointer_type; // ?
    typedef detail::strided_base_cursor<const Value*> super;
    typedef strided_dense_el_cursor       self;

    //  strided_dense_el_cursor () {} 
    strided_dense_el_cursor (const_pointer_type me, size_t stride) : super(me, stride) {}

    template <typename Parameters>
    strided_dense_el_cursor(dense2D<Value, Parameters> const& ma, size_t r, size_t c, size_t stride)
	: super(ma.elements() + ma.indexer(ma, r, c), stride)
    {}

    // Why do we need this?
    strided_dense_el_cursor(super const& x) : super(x) {}

    self operator+(int x) const
    {
	return super::operator+(x);
    }
};


// Iterator over every element (contiguous memory)
// Simply a pointer 
template <typename Value> 
typedef *Value dense_el_iterator;

// Same with const
template <typename Value> 
typedef const *Value dense_el_const_iterator;


// Strided iterator *operator returns (const) reference to Value instead of key
// row(i) and col(i) doesn't work
template <typename Value>
struct strided_dense_el_const_iterator
    : public detail::strided_base_cursor<const Value*> 
{
    typedef const value_type*                         key_type;
    typedef detail::strided_base_cursor<key_type>     super;
    typedef strided_dense_el_const_iterator           self;

    strided_dense_el_const_iterator(key_type me, size_t stride) : super(me, stride) {}

    template <typename Parameters>
    strided_dense_el_const_iterator(dense2D<Value, Parameters> const& ma, size_t r, size_t c, size_t stride)
	: super(ma.elements() + ma.indexer(ma, r, c), stride)
    {}

    self operator+(int x) const
    {
	return super::operator+(x);
    }

    const Value& operator*() const
    {
	return *(this->key);
    }
};


template <typename Value>
struct strided_dense_el_const_iterator
    : public detail::strided_base_cursor<Value*> 
{
    typedef value_type*                               key_type;
    typedef detail::strided_base_cursor<key_type>     super;
    typedef strided_dense_el_const_iterator           self;

    strided_dense_el_const_iterator(key_type me, size_t stride) : super(me, stride) {}

    template <typename Parameters>
    strided_dense_el_const_iterator(dense2D<Value, Parameters>& ma, size_t r, size_t c, size_t stride)
	: super(ma.elements() + ma.indexer(ma, r, c), stride)
    {}

    self operator+(int x) const
    {
	return super::operator+(x);
    }

    Value& operator*() const
    {
	return *(this->key);
    }
};



// Indexing for dense matrices
struct dense2D_indexer 
{
  private:
    // helpers for public functions
    size_t offset(size_t ldim, size_t r, size_t c, row_major) const 
    {
	return r * ldim + c; 
    }
    size_t offset(size_t ldim, size_t r, size_t c, col_major) const 
    {
	return c * ldim + r; 
    }
    
    size_t row(size_t offset, size_t ldim, row_major) const 
    {
	return offset / ldim; 
    }
    size_t row(size_t offset, size_t ldim, col_major) const 
    {
	return offset % ldim;
    }
    
    size_t col(size_t offset, size_t ldim, row_major) const 
    {
	return offset % ldim;
    }
    size_t col(size_t offset, size_t ldim, col_major) const 
    {
	return offset / ldim; 
    }

 public:
    template <typename Value, class Parameters>
    size_t operator() (const dense2D<Value, Parameters>& ma, size_t r, size_t c) const
    {
	typedef dense2D<Value, Parameters> matrix_type;
	// convert into c indices
	typename matrix_type::index_type my_index;
	size_t my_r= index::change_from(my_index, r);
	size_t my_c= index::change_from(my_index, c);
	return offset(ma.ldim, my_r, my_c, typename matrix_type::orientation());
    }

    template <typename Value, class Parameters>
    size_t row(const dense2D<Value, Parameters>& ma, 
	       typename dense2D<Value, Parameters>::key_type key) const
    {
	typedef dense2D<Value, Parameters> matrix_type;
	// row with c-index for my orientation
	size_t r= row(ma.offset(key), ma.ldim, typename matrix_type::orientation());
	return index::change_to(typename matrix_type::index_type(), r);
    }

    template <typename Value, class Parameters>
    size_t col(const dense2D<Value, Parameters>& ma, 
	       typename dense2D<Value, Parameters>::key_type key) const 
    {
	typedef dense2D<Value, Parameters> matrix_type;
	// column with c-index for my orientation
	size_t c= col(ma.offset(key), ma.ldim, typename matrix_type::orientation());
	return index::change_to(typename matrix_type::index_type(), c);
    }
    template <typename, typename> friend struct dense2D;
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
template <typename Value, typename Parameters = mtl::matrix_parameters<> >
class dense2D : public detail::base_sub_matrix<Value, Parameters>, 
		public detail::contiguous_memory_matrix< Value, Parameters::on_stack, 
							 detail::dense2D_array_size<Parameters, Parameters::on_stack>::value >,
                public detail::crtp_base_matrix< dense2D<Value, Parameters>, Value, std::size_t >
{
    typedef dense2D                                    self;
    typedef detail::base_sub_matrix<Value, Parameters>   super;
  public:
    typedef Parameters                        parameters;
    typedef typename Parameters::orientation  orientation;
    typedef typename Parameters::index        index_type;
    typedef typename Parameters::dimensions   dim_type;
    typedef Value                               value_type;

    typedef detail::contiguous_memory_matrix<Value, Parameters::on_stack, 
					     detail::dense2D_array_size<Parameters, Parameters::on_stack>::value>     super_memory;

    typedef const value_type*                 const_pointer_type;
    typedef const_pointer_type                key_type;
    typedef std::size_t                       size_type;
    typedef dense_el_cursor<Value>              el_cursor_type;  
    typedef dense2D_indexer                   indexer_type;

    // Self-similar type unless dimension is fixed
    // Not supported for the moment
    typedef self                              sub_matrix_type;  

  protected:
    // Obviously, the next 3 functions must be called after setting dimensions
    void set_nnz()
    {
	this->my_nnz = this->num_rows() * this->num_cols();
    }

    void set_ldim(row_major)
    {
	ldim= this->num_cols();
    }

    void set_ldim(col_major)
    {
	ldim= this->num_rows();
    }

    void set_ldim()
    {
	set_ldim(orientation());
    }

  public:
    // if compile time matrix size allocate memory
    dense2D() : super(), super_memory(dim_type().num_rows() * dim_type().num_cols()) 
    {
	set_nnz(); set_ldim();
    }

    // only sets dimensions, only for run-time dimensions
    explicit dense2D(mtl::non_fixed::dimensions d) 
	: super(d), super_memory(d.num_rows() * d.num_cols()) 
    {
	set_nnz(); set_ldim();
    }

    dense2D(size_type num_rows, size_type num_cols) 
	: super(mtl::non_fixed::dimensions(num_rows, num_cols)), 
	  super_memory(num_rows * num_cols) 
    {
	set_nnz(); set_ldim();
    }

    // sets dimensions and pointer to external data
    explicit dense2D(mtl::non_fixed::dimensions d, value_type* a) 
      : super(d), super_memory(a, d.num_rows() * d.num_cols()) 
    { 
        set_nnz(); set_ldim();
    }

    // same constructor for compile time matrix size
    // sets dimensions and pointer to external data
    explicit dense2D(value_type* a) : super(), super_memory(a) 
    { 
	BOOST_STATIC_ASSERT((dim_type::is_static));
        set_nnz(); set_ldim();
    }

    // old style, better use value property map
    value_type operator() (size_t r, size_t c) const 
    {
	size_t offset= indexer(*this, r, c);
        return this->data[offset];
    }

    value_type& operator() (size_t r, size_t c)
    {
	return this->data[indexer(*this, r, c)]; 
    }    

    // offset regarding c-style indices
    size_t c_offset(size_t r, size_t c)
    {
	return indexer.offset(ldim, r, c, orientation());
    }

  protected:
    
    // Set ranges from begin_r to end_r and begin_c to end_c
    void set_ranges(size_type begin_r, size_type end_r, size_type begin_c, size_type end_c)
    {
	super::set_ranges(begin_r, end_r, begin_c, end_c);
	set_nnz();
    }
	
    // Set ranges to a num_row x num_col matrix, keeps indexing
    void set_ranges(size_type num_rows, size_type num_cols)
    {
	set_ranges(this->begin_row(), this->begin_row() + num_rows, 
		   this->begin_col(), this->begin_col() + num_cols);
    }
    

  public:

    indexer_type  indexer;
  protected:
    // Leading dimension is minor dimension in original matrix 
    // Opposed to other dims doesn't change in sub-matrices
    size_type     ldim; 

    friend class dense2D_indexer;
    template <typename> friend struct sub_matrix_t;
    template <typename, typename> friend struct traits::range_generator;
}; // dense2D



// =============
// Property Maps
// =============

namespace traits 
{
    template <typename Value, class Parameters>
    struct row<dense2D<Value, Parameters> >
    {
        typedef mtl::detail::indexer_row_ref<dense2D<Value, Parameters> > type;
    };

    template <typename Value, class Parameters>
    struct col<dense2D<Value, Parameters> >
    {
        typedef mtl::detail::indexer_col_ref<dense2D<Value, Parameters> > type;
    };

    template <typename Value, class Parameters>
    struct const_value<dense2D<Value, Parameters> >
    {
        typedef mtl::detail::direct_const_value<dense2D<Value, Parameters> > type;
    };

    template <typename Value, class Parameters>
    struct value<dense2D<Value, Parameters> >
    {
        typedef mtl::detail::direct_value<dense2D<Value, Parameters> > type;
    };

    template <typename Value, class Parameters>
    struct is_mtl_type<dense2D<Value, Parameters> > 
    {
	static bool const value= true; 
    };

    // define corresponding type without all template parameters
    template <typename Value, class Parameters>
    struct matrix_category<dense2D<Value, Parameters> > 
    {
	typedef tag::dense2D type;
    };

} // namespace traits
    

// ================
// Range generators
// ================

namespace traits
{
    template <typename Value, class Parameters>
    struct range_generator<glas::tags::all_t, dense2D<Value, Parameters> >
      : detail::dense_element_range_generator<dense2D<Value, Parameters>,
					      dense_el_cursor<Value>, complexity_classes::linear_cached>
    {};

    template <typename Value, class Parameters>
    struct range_generator<glas::tags::nz_t, dense2D<Value, Parameters> >
      : detail::dense_element_range_generator<dense2D<Value, Parameters>,
					      dense_el_cursor<Value>, complexity_classes::linear_cached>
    {};

    namespace detail 
    {
	// complexity of dense row cursor depends on storage scheme
	// if orientation is row_major then complexity is cached_linear, otherwise linear
	template <typename Orientation> struct dense2D_rc {};
	template<> struct dense2D_rc<row_major>
	{
	    typedef complexity_classes::linear_cached type;
	};
	template<> struct dense2D_rc<col_major>
	{
	    typedef complexity_classes::linear type;
	};

	// Complexity of column cursor is of course opposite
	template <typename Orientation> struct dense2D_cc
	    : dense2D_rc<typename transposed_orientation<Orientation>::type>
	{};
    }

    template <typename Value, class Parameters>
    struct range_generator<glas::tags::row_t, dense2D<Value, Parameters> >
	: detail::all_rows_range_generator<dense2D<Value, Parameters>, 
					   typename detail::dense2D_rc<typename Parameters::orientation>::type>
    {};
 
    // For a cursor pointing to some row give the range of elements in this row 
    template <typename Value, class Parameters>
    struct range_generator<glas::tags::nz_t, 
			   detail::sub_matrix_cursor<dense2D<Value, Parameters>, glas::tags::row_t, 2> >
    {
	typedef dense2D<Value, Parameters>  matrix;
	typedef detail::sub_matrix_cursor<matrix, glas::tags::row_t, 2> cursor;
	// linear for col_major and linear_cached for row_major
	typedef typename detail::dense2D_rc<typename Parameters::orientation>::type   complexity;
	static int const             level = 1;
	// for row_major dense_el_cursor would be enough, i.e. bit less overhead but uglier code
	typedef strided_dense_el_cursor<Value> type;
	size_t stride(cursor const&, row_major)
	{
	    return 1;
	}
	size_t stride(cursor const& c, col_major)
	{
	    return c.ref.ldim;
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

    template <typename Value, class Parameters>
    struct range_generator<glas::tags::all_t, 
			   detail::sub_matrix_cursor<dense2D<Value, Parameters>, glas::tags::row_t, 2> >
        : range_generator<glas::tags::nz_t, 
			  detail::sub_matrix_cursor<dense2D<Value, Parameters>, glas::tags::row_t, 2> >
    {};



    template <typename Value, class Parameters>
    struct range_generator<glas::tags::col_t, dense2D<Value, Parameters> >
	: detail::all_cols_range_generator<dense2D<Value, Parameters>, 
					   typename detail::dense2D_cc<typename Parameters::orientation>::type>
    {};
 
    // For a cursor pointing to some row give the range of elements in this row 
    template <typename Value, class Parameters>
    struct range_generator<glas::tags::nz_t, 
			   detail::sub_matrix_cursor<dense2D<Value, Parameters>, glas::tags::col_t, 2> >
    {
	typedef dense2D<Value, Parameters>  matrix;
	typedef detail::sub_matrix_cursor<matrix, glas::tags::col_t, 2> cursor;	
	typedef typename detail::dense2D_cc<typename Parameters::orientation>::type   complexity;
	static int const             level = 1;
	typedef strided_dense_el_cursor<Value> type;
	size_t stride(cursor const&, col_major)
	{
	    return 1;
	}
	size_t stride(cursor const& c, row_major)
	{
	    return c.ref.ldim;
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

    template <typename Value, class Parameters>
    struct range_generator<glas::tags::all_t, 
			   detail::sub_matrix_cursor<dense2D<Value, Parameters>, glas::tags::col_t, 2> >
        : range_generator<glas::tags::nz_t, 
			  detail::sub_matrix_cursor<dense2D<Value, Parameters>, glas::tags::col_t, 2> >
    {};



} // namespace traits


// ==========
// Sub matrix
// ==========

template <typename Value, typename Parameters>
struct sub_matrix_t<dense2D<Value, Parameters> >
{
    typedef dense2D<Value, Parameters>        matrix_type;
    typedef matrix_type                     sub_matrix_type;
    typedef matrix_type const               const_sub_matrix_type;
    typedef typename matrix_type::size_type size_type;
    
    sub_matrix_type operator()(matrix_type& matrix, size_type begin_r, size_type end_r, size_type begin_c, size_type end_c)
    {
	matrix.check_ranges(begin_r, end_r, begin_c, end_c);

	sub_matrix_type  tmp(matrix);

	// Leading dimension doesn't change
	tmp.data += matrix.indexer(matrix, begin_r, begin_c);  // Takes care of indexing
	tmp.set_ranges(end_r - begin_r, end_c - begin_c);

	// sub matrix doesn't own the memory (and must not free at the end)
	tmp.extern_memory= true;

	return tmp;
    }

    const_sub_matrix_type
    operator()(matrix_type const& matrix, size_type begin_r, size_type end_r, size_type begin_c, size_type end_c)
    {
	// To minimize code duplication, we use the non-const version
	sub_matrix_type tmp((*this)(const_cast<matrix_type&>(matrix), begin_r, end_r, begin_c, end_c));
	return tmp;
    }	
};

} // namespace mtl

#endif // MTL_DENSE2D_INCLUDE


/*
Limitations:
- with compile-time constant dimension, submatrices are not supported (would violate self-similarity)
- Element cursor doesn't work for sub-matrices (not contiguous)
*/
