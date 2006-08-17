// $COPYRIGHT$

#ifndef MTL_MORTON_DENSE_INCLUDE
#define MTL_MORTON_DENSE_INCLUDE

#include <boost/numeric/mtl/common_includes.hpp>
#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/detail/base_sub_matrix.hpp>
#include <boost/numeric/mtl/detail/contiguous_memory_matrix.hpp>
#include <boost/numeric/mtl/detail/dilated_int.hpp>
// #include <boost/numeric/mtl/ahnentafel_index.hpp>
// #include <vector>

namespace mtl {

template <std::size_t  BitMask>
struct morton_dense_key
{
    typedef std::size_t                               size_type;
    typedef dilated_int<std::size_t, BitMask, true>   dilated_row_t;
    typedef dilated_int<std::size_t, ~BitMask, true>  dilated_col_t; 
    typedef morton_dense_key                          self;

    morton_dense_key(size_type my_row, size_type my_col) 
	: my_row(my_row), my_col(my_col), dilated_row(my_row), dilated_col(my_col)
    {}

    bool operator== (self const& x)
    {
	return my_row == x.my_row && my_col == x.my_col;
    }

    bool operator!= (self const& x)
    {
	return !(*this == x);
    }

    size_type row() const
    {
	return my_row;
    }

    size_type col() const
    {
	return my_col;
    }

public:
    size_type                       my_row, my_col;   
    dilated_row_t                dilated_row;
    dilated_col_t                dilated_col; 
};

template <std::size_t  BitMask>
struct morton_dense_el_cursor 
    : public morton_dense_key<BitMask>
{
    typedef std::size_t                               size_type;
    typedef dilated_int<std::size_t, ~BitMask, true>  dilated_col_t; 
    typedef morton_dense_el_cursor                    self;
    typedef morton_dense_key<BitMask>                 base;

    morton_dense_el_cursor(size_type my_row, size_type my_col, size_type num_cols) 
	: base(my_row, my_col), num_cols(num_cols) 
    {}

    self& operator++ ()
    {
	++this->my_col; ++this->dilated_col;
	if (this->my_col == num_cols) {
	    this->my_col= 0; this->dilated_col= dilated_col_t(0);
	    ++this->my_row; ++this->dilated_row;
	}
	return *this;
    }

    base& operator* ()
    {
	return *this;
    }

protected:
    size_t                       num_cols;
};

template <std::size_t  BitMask>
struct morton_dense_row_cursor 
    : public morton_dense_key<BitMask>
{
    typedef std::size_t                               size_type;
    typedef morton_dense_row_cursor                   self;
    typedef morton_dense_key<BitMask>                 base;

    morton_dense_row_cursor(size_type my_row, size_type my_col) 
	: base(my_row, my_col)
    {}

    self& operator++ ()
    {
	++this->my_row; ++this->dilated_row;
	return *this;
    }

    base& operator* ()
    {
	return *this;
    }
};

template <std::size_t  BitMask>
struct morton_dense_col_cursor 
    : public morton_dense_key<BitMask>
{
    typedef std::size_t                               size_type;
    typedef morton_dense_col_cursor                   self;
    typedef morton_dense_key<BitMask>                 base;

    morton_dense_col_cursor(size_type my_row, size_type my_col) 
	: base(my_row, my_col)
    {}

    self& operator++ ()
    {
	++this->my_col; ++this->dilated_col;
	return *this;
    }

    base& operator* ()
    {
	return *this;
    }
};





// Morton Dense matrix type
template <typename Elt, std::size_t  BitMask, typename Parameters>
class morton_dense : public detail::base_sub_matrix<Elt, Parameters>, 
		     public detail::contiguous_memory_matrix<Elt, false>
{
    typedef detail::base_sub_matrix<Elt, Parameters>            super;
    typedef detail::contiguous_memory_matrix<Elt, false>        super_memory;
    typedef morton_dense                                        self;

  public:

    typedef typename Parameters::orientation  orientation;
    typedef typename Parameters::index        index_type;
    typedef typename Parameters::dimensions   dim_type;
    typedef Elt                               value_type;
    typedef std::size_t                       size_type;

    // typedef self                              sub_matrix_type;
    // typedef morton_dense_el_cursor<Elt>       el_cursor_type;  
    // typedef morton_dense_indexer              indexer_type;

    //  implement cursor for morton matrix, somewhere
    //  also, morton indexer?

    typedef morton_dense_key<BitMask>          key_type;
    typedef morton_dense_el_cursor<BitMask>    el_cursor_type; 
    
    typedef dilated_int<std::size_t, BitMask, true>   dilated_row_t;
    typedef dilated_int<std::size_t, ~BitMask, true>  dilated_col_t; 
    //  typedef morton_dense_indexer              indexer_type;


  public: 

#if 0
    // All Ahnentafel stuff is commented out
    // add morton functions here

    // boundary checking of the Ahnentafel index

    bool isRoot(const AhnenIndex& index) const;
    bool isLeaf(const AhnenIndex& index) const;
    bool isInBound(const AhnenIndex& index) const;


    // matrix access functions

    int getRows() const;
    int getCols() const;
    int getMaxLevel() const;
    int getLevel(const AhnenIndex& index) const;
    int getBlockOrder(const AhnenIndex& index) const;
    int getBlockSize(const AhnenIndex& index) const;
    // get the value of the matrix element corresponding
    // to the Ahnentafel index
    value_type getElement(const AhnenIndex& index) const;

    int getRowMask() const;
    int getColMask() const;

    // debugging functions
    void printVec() const;
    void printMat() const;
#endif

  protected:
    
    // ranges of rows and columns
    dilated_row_t            my_begin_row, my_end_row;
    dilated_col_t            my_begin_col, my_end_col;

    // Set ranges from begin_r to end_r and begin_c to end_c
    void set_ranges(size_type begin_r, size_type end_r, size_type begin_c, size_type end_c)
    {
	super::set_ranges(begin_r, end_r, begin_c, end_c);
	my_begin_row= begin_r; my_end_row= end_r;
	my_begin_col= begin_c; my_end_col= end_c;
	set_nnz();
    }

    // Set ranges to a num_row x num_col matrix, keeps indexing
    void set_ranges(size_type num_rows, size_type num_cols)
    {
	set_ranges(this->begin_row(), this->begin_row() + num_rows, 
		   this->begin_col(), this->begin_col() + num_cols);
    }

  public:
    // if compile time matrix size allocate memory
    morton_dense() : super_memory( memory_need( dim_type().num_rows(), dim_type().num_cols() ) ) 
    {
	set_ranges(dim_type().num_rows(), dim_type().num_cols());
    }

    // only sets dimensions, only for run-time dimensions
    explicit morton_dense(mtl::non_fixed::dimensions d) 
	: super_memory( memory_need( d.num_rows(), d.num_cols() ) ) 
    {
	// set_nnz();
	set_ranges(d.num_rows(), d.num_cols());
    }

    // sets dimensions and pointer to external data
    explicit morton_dense(mtl::non_fixed::dimensions d, value_type* a) 
      : super_memory(a) 
    { 
        // set_nnz();
	set_ranges(d.num_rows(), d.num_cols());
    }

    // same constructor for compile time matrix size
    // sets dimensions and pointer to external data
    explicit morton_dense(value_type* a) : super_memory(a) 
    { 
	BOOST_ASSERT((dim_type::is_static));
	set_ranges(dim_type().num_rows(), dim_type().num_cols());
    }

    value_type operator() (key_type const& key) const
    {
	return this->data[key.dilated_row.dilated_value() + key.dilated_col.dilated_value()];
    }

    void operator()(key_type const& key, value_type const& value)
    {
	this->data[key.dilated_row.dilated_value() + key.dilated_col.dilated_value()]= value;
    }

  protected:
    void set_nnz()
    {
      this->my_nnz = this->num_rows() * this->num_cols();
    }
    
    size_type memory_need(size_type rows, size_type cols)
    {
        dilated_row_t n_rows(rows - 1);
        dilated_col_t n_cols(cols - 1);
        return (n_rows.dilated_value() + n_cols.dilated_value() + 1);
    }


    template <typename> friend struct sub_matrix_t;    
    
#if 0
  private:
  // add morton member variables here

    int rows_;          // number of rows of the matrix
    int cols_;          // number of columns of the matrix
    int quadOrder_;     // order of the level-0 quad
                        // quadOrder_ = pow(2, (int)log2(max(rows_, cols_))) + 1)
                        // or quadOrder = pow(2, (int)log2(max(rows_, cols_)))),
                        // depending on the value of rows_ and cols_.
    int storageSize_;   // size of allocated storage for the matrix
    int maxLevel_;      // maximum level of the quadtree for the matrix

    std::vector<int> upBoundVec_;    // upper boundary vector
    std::vector<int> lowBoundVec_;   // lower boundary vector
    std::vector<int> rowMaskVec_;    // row mask vector
    std::vector<int> colMaskVec_;    // col mask vector
    // T* data_;                   // a pointer to the matrix data array

    void setQuadOrder();        // set quadOrder_
    void setStorageSize();      // set storageSize_
    void setMaxLevel();         // set maxLevel_
    void setBoundVec();         // set boundary vectors
    void setMaskVec();          // set mask vectors
    void mkMortonSPDMatrix();   // make default Morton matrix
#endif
};


#if 0
// boundary checking of Ahnentafel index

// check if the index is the root
template <typename Elt, std::size_t  BitMask, typename Parameters>
bool morton_dense<Elt, BitMask, Parameters>::isRoot(const AhnenIndex& index) const {
    return index.getIndex() == 3;
}

// check if the index is a leaf
template <typename Elt, std::size_t  BitMask, typename Parameters>
bool morton_dense<Elt, BitMask, Parameters>::isLeaf(const AhnenIndex& index) const {
  // a possible better way: compare index with the boundary
  // vector directly, instead of calling isInBound()
    if(isInBound(index))
        return (index.getIndex() >= lowBoundVec_[maxLevel_]);
    else return 0;
}
#endif



// =============
// Property Maps
// =============

namespace traits
{
  template <class Elt, std::size_t  BitMask, class Parameters>
  struct row<morton_dense<Elt, BitMask, Parameters> >
  {
    typedef mtl::detail::row_in_key<morton_dense<Elt, BitMask, Parameters> > type;
  };

  template <class Elt, std::size_t  BitMask, class Parameters>
  struct col<morton_dense<Elt, BitMask, Parameters> >
  {
    typedef mtl::detail::col_in_key<morton_dense<Elt, BitMask, Parameters> > type;
  };

  template <class Elt, std::size_t  BitMask, class Parameters>
  struct const_value<morton_dense<Elt, BitMask, Parameters> >
  {
    typedef mtl::detail::matrix_const_value_ref<morton_dense<Elt, BitMask, Parameters> > type;
  };

  template <class Elt, std::size_t  BitMask, class Parameters>
  struct value<morton_dense<Elt, BitMask, Parameters> >
  {
    typedef mtl::detail::matrix_value_ref<morton_dense<Elt, BitMask, Parameters> > type;
  };

  template <class Elt, std::size_t  BitMask, class Parameters>
  struct is_mtl_type<morton_dense<Elt, BitMask, Parameters> >
  {
    static bool const value= true;
  };

  // define corresponding type without all template parameters
  template <class Elt, std::size_t  BitMask, class Parameters>
  struct matrix_category<morton_dense<Elt, BitMask, Parameters> >
  {
    typedef mtl::tag::morton_dense type;
  };

} // namespace traits


// Range generators
// ================

namespace traits
{

    template <class Elt, std::size_t  BitMask, class Parameters>
    struct range_generator<glas::tags::all_t, morton_dense<Elt, BitMask, Parameters> >
    {
	typedef morton_dense<Elt, BitMask, Parameters>        Matrix;
	typedef complexity::linear_cached        complexity;
	static int const                         level = 1;
	typedef morton_dense_el_cursor<BitMask>  type;
	type begin(Matrix const& matrix)
	{
	    return type(matrix.begin_row(), matrix.begin_col(), matrix.num_cols());
	}
	type end(Matrix const& matrix)
	{
	    return type(matrix.end_row(), matrix.begin_col(), matrix.num_cols());
	}
    };

    template <class Elt, std::size_t  BitMask, class Parameters>
    struct range_generator<glas::tags::nz_t, morton_dense<Elt, BitMask, Parameters> >
	: range_generator<glas::tags::all_t, morton_dense<Elt, BitMask, Parameters> >
    {};

    template <class Elt, std::size_t  BitMask, class Parameters>
    struct range_generator<glas::tags::row_t, morton_dense<Elt, BitMask, Parameters> >
	: detail::all_rows_range_generator<morton_dense<Elt, BitMask, Parameters>, complexity::linear_cached> 
    {};

    // For a cursor pointing to some row give the range of elements in this row 
    template <class Elt, std::size_t  BitMask, class Parameters>
    struct range_generator<glas::tags::nz_t, 
			   detail::sub_matrix_cursor<morton_dense<Elt, BitMask, Parameters>, glas::tags::row_t, 2> >
    {
	typedef morton_dense<Elt, BitMask, Parameters>                   matrix;
	typedef detail::sub_matrix_cursor<matrix, glas::tags::row_t, 2>  cursor;
	typedef complexity::linear_cached                                complexity;
	static int const                                                 level = 1;
	typedef morton_dense_col_cursor<BitMask>                         type;
	
	type begin(cursor const& c)
	{
	    return type(c.key, c.ref.begin_col());
	}
	type end(cursor const& c)
	{
	    return type(c.key, c.ref.end_col());
	}
    };

    template <class Elt, std::size_t  BitMask, class Parameters>
    struct range_generator<glas::tags::all_t, 
			   detail::sub_matrix_cursor<morton_dense<Elt, BitMask, Parameters>, glas::tags::row_t, 2> >
        : range_generator<glas::tags::nz_t, 
			  detail::sub_matrix_cursor<morton_dense<Elt, BitMask, Parameters>, glas::tags::row_t, 2> >
    {};

    template <class Elt, std::size_t  BitMask, class Parameters>
    struct range_generator<glas::tags::col_t, morton_dense<Elt, BitMask, Parameters> >
	: detail::all_cols_range_generator<morton_dense<Elt, BitMask, Parameters>, complexity::linear_cached> 
    {};

    // For a cursor pointing to some row give the range of elements in this row 
    template <class Elt, std::size_t  BitMask, class Parameters>
    struct range_generator<glas::tags::nz_t, 
			   detail::sub_matrix_cursor<morton_dense<Elt, BitMask, Parameters>, glas::tags::col_t, 2> >
    {
	typedef morton_dense<Elt, BitMask, Parameters>                   matrix;
	typedef detail::sub_matrix_cursor<matrix, glas::tags::col_t, 2>  cursor;
	typedef complexity::linear_cached                                complexity;
	static int const                                                 level = 1;
	typedef morton_dense_row_cursor<BitMask>                         type;
	
	type begin(cursor const& c)
	{
	    return type(c.ref.begin_row(), c.key);
	}
	type end(cursor const& c)
	{
	    return type(c.ref.end_row(), c.key);
	}
    };

    template <class Elt, std::size_t  BitMask, class Parameters>
    struct range_generator<glas::tags::all_t, 
			   detail::sub_matrix_cursor<morton_dense<Elt, BitMask, Parameters>, glas::tags::col_t, 2> >
        : range_generator<glas::tags::nz_t, 
			  detail::sub_matrix_cursor<morton_dense<Elt, BitMask, Parameters>, glas::tags::col_t, 2> >
    {};
} // namespace traits


// ==========
// Sub matrix
// ==========

template <typename Elt, std::size_t  BitMask, typename Parameters>
struct sub_matrix_t<morton_dense<Elt, BitMask, Parameters> >
{
    typedef morton_dense<Elt, BitMask, Parameters>    matrix_type;
    typedef matrix_type                     sub_matrix_type;
    typedef matrix_type const               const_sub_matrix_type;
    typedef typename matrix_type::size_type size_type;
    
    sub_matrix_type operator()(matrix_type& matrix, size_type begin_r, size_type end_r, size_type begin_c, size_type end_c)
    {
	matrix.check_ranges(begin_r, end_r, begin_c, end_c);
	// Probably check whether power of 2 is crossed (ask David and Michael)

	sub_matrix_type  tmp(matrix);

	typename matrix_type::dilated_row_t  dilated_row(begin_r);
	typename matrix_type::dilated_col_t  dilated_col(begin_c);

	// Set new start address within masked matrix
	tmp.data += dilated_row.dilated_value() + dilated_col.dilated_value();
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

#endif // MTL_MORTON_DENSE_INCLUDE
