// $COPYRIGHT$

#ifndef MTL_MORTON_DENSE_INCLUDE
#define MTL_MORTON_DENSE_INCLUDE

#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>

#include <boost/numeric/mtl/utility/common_include.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/detail/base_sub_matrix.hpp>
#include <boost/numeric/mtl/detail/contiguous_memory_matrix.hpp>
#include <boost/numeric/mtl/detail/dilated_int.hpp>
#include <boost/numeric/mtl/utility/iterator_adaptor.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>


// #include <boost/numeric/mtl/ahnentafel_detail/index.hpp>

namespace mtl {

template <unsigned long BitMask>
struct morton_dense_key
{
    typedef std::size_t                               size_type;
    typedef dilated_int<std::size_t, BitMask, true>   dilated_row_t;
    typedef dilated_int<std::size_t, ~BitMask, true>  dilated_col_t; 
    typedef morton_dense_key                          self;

    morton_dense_key(size_type my_row, size_type my_col) 
	: my_row(my_row), my_col(my_col), dilated_row(my_row), dilated_col(my_col)
    {}

    bool operator== (self const& x) const
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

    self& advance_row(int row_inc)
    {
	dilated_row.advance(row_inc);
	// potential addition of signed and unsigned
	my_row+= row_inc;
	return *this;
    }

    self& advance_col(int col_inc)
    {
	dilated_col.advance(col_inc);
	// potential addition of signed and unsigned
	my_col+= col_inc;
	return *this;
    }

    self& advance(int row_inc, int col_inc)
    {
	advance_row(row_inc);
	advance_col(col_inc);
	return *this;
    }

public:
    size_type                    my_row, my_col;   
    dilated_row_t                dilated_row;
    dilated_col_t                dilated_col; 
};

template <unsigned long BitMask>
struct morton_dense_el_cursor 
    : public morton_dense_key<BitMask>
{
    typedef std::size_t                               size_type;
    typedef dilated_int<std::size_t, ~BitMask, true>  dilated_col_t; 
    typedef morton_dense_el_cursor                    self;
    typedef morton_dense_key<BitMask>                 base;
    typedef base                                      key_type;

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

    const base& operator* () const
    {
	return *this;
    }

protected:
    size_t                       num_cols;
};

template <unsigned long BitMask>
struct morton_dense_row_cursor 
    : public morton_dense_key<BitMask>
{
    typedef std::size_t                               size_type;
    typedef morton_dense_row_cursor                   self;
    typedef morton_dense_key<BitMask>                 base;
    typedef base                                      key_type;

    morton_dense_row_cursor(size_type my_row, size_type my_col) 
	: base(my_row, my_col)
    {}

    self& operator++ ()
    {
	++this->my_row; ++this->dilated_row;
	return *this;
    }

    self& operator+=(int inc) 
    {
	this->advance_row(inc);
	return *this;
    };

    self operator+ (int inc) const
    {
	self tmp(*this);
	tmp.advance_row(inc);
	return tmp;
    }

    base& operator* ()
    {
	return *this;
    }

    const base& operator* () const
    {
	return *this;
    }
};

template <unsigned long BitMask>
struct morton_dense_col_cursor 
    : public morton_dense_key<BitMask>
{
    typedef std::size_t                               size_type;
    typedef morton_dense_col_cursor                   self;
    typedef morton_dense_key<BitMask>                 base;
    typedef base                                      key_type;

    morton_dense_col_cursor(size_type my_row, size_type my_col) 
	: base(my_row, my_col)
    {}

    self& operator++ ()
    {
	++this->my_col; ++this->dilated_col;
	return *this;
    }

    self& operator+=(int inc) 
    {
	this->advance_col(inc);
	return *this;
    };

    self operator+ (int inc) const
    {
	self tmp(*this);
	tmp.advance_col(inc);
	return tmp;
    }

    base& operator* ()
    {
	return *this;
    }

    const base& operator* () const
    {
	return *this;
    }
};


template <typename Matrix>
struct morton_dense_row_const_iterator
    : utilities::const_iterator_adaptor<typename traits::const_value<Matrix>::type, morton_dense_row_cursor<Matrix::mask>,
					typename Matrix::value_type>
{
    static const unsigned long                          mask= Matrix::mask;
    typedef morton_dense_row_cursor<mask>               cursor_type;
    typedef typename traits::const_value<Matrix>::type  map_type;
    typedef typename Matrix::value_type                 value_type;
    typedef typename Matrix::size_type                  size_type;
    typedef utilities::const_iterator_adaptor<map_type, cursor_type, value_type> base;
    
    morton_dense_row_const_iterator(const Matrix& matrix, size_type row, size_type col)
	: base(map_type(matrix), cursor_type(row, col))
    {}
};


template <typename Matrix>
struct morton_dense_row_iterator
    : utilities::iterator_adaptor<typename traits::value<Matrix>::type, morton_dense_row_cursor<Matrix::mask>,
				  typename Matrix::value_type>
{
    static const unsigned long                          mask= Matrix::mask;
    typedef morton_dense_row_cursor<mask>               cursor_type;
    typedef typename traits::value<Matrix>::type        map_type;
    typedef typename Matrix::value_type                 value_type;
    typedef typename Matrix::size_type                  size_type;
    typedef utilities::iterator_adaptor<map_type, cursor_type, value_type> base;
    
    morton_dense_row_iterator(Matrix& matrix, size_type row, size_type col)
	:  base(map_type(matrix), cursor_type(row, col))
    {}
};


template <typename Matrix>
struct morton_dense_col_const_iterator
    : utilities::const_iterator_adaptor<typename traits::const_value<Matrix>::type, morton_dense_col_cursor<Matrix::mask>,
					typename Matrix::value_type>
{
    static const unsigned long                          mask= Matrix::mask;
    typedef morton_dense_col_cursor<mask>               cursor_type;
    typedef typename traits::const_value<Matrix>::type  map_type;
    typedef typename Matrix::value_type                 value_type;
    typedef typename Matrix::size_type                  size_type;
    typedef utilities::const_iterator_adaptor<map_type, cursor_type, value_type> base;
    
    morton_dense_col_const_iterator(const Matrix& matrix, size_type row, size_type col)
	: base(map_type(matrix), cursor_type(row, col))
    {}
};


template <typename Matrix>
struct morton_dense_col_iterator
    : utilities::iterator_adaptor<typename traits::value<Matrix>::type, morton_dense_col_cursor<Matrix::mask>,
				  typename Matrix::value_type>
{
    static const unsigned long                          mask= Matrix::mask;
    typedef morton_dense_col_cursor<mask>               cursor_type;
    typedef typename traits::value<Matrix>::type        map_type;
    typedef typename Matrix::value_type                 value_type;
    typedef typename Matrix::size_type                  size_type;
    typedef utilities::iterator_adaptor<map_type, cursor_type, value_type> base;
    
    morton_dense_col_iterator(Matrix& matrix, size_type row, size_type col)
      : base(map_type(matrix), cursor_type(row, col))  {}    
};



// Morton Dense matrix type 
template <typename Elt, unsigned long BitMask, typename Parameters = mtl::matrix_parameters<> >
class morton_dense : public detail::base_sub_matrix<Elt, Parameters>, 
		     public detail::contiguous_memory_matrix<Elt, false>,
                     public detail::crtp_base_matrix< morton_dense<Elt, BitMask, Parameters>, Elt, std::size_t >
{
    typedef detail::base_sub_matrix<Elt, Parameters>            super;
    typedef detail::contiguous_memory_matrix<Elt, false>        super_memory;
    typedef morton_dense                                        self;

  public:

    typedef Parameters                        parameters;
    typedef typename Parameters::orientation  orientation;
    typedef typename Parameters::index        index_type;
    typedef typename Parameters::dimensions   dim_type;
    typedef Elt                               value_type;
    // return type of operator() const
    typedef const value_type&                 const_access_type;
    typedef std::size_t                       size_type;
    const static size_type                    mask= BitMask;

    // typedef self                              sub_matrix_type;
    // typedef morton_dense_el_cursor<Elt>       el_cursor_type;  
    // typedef morton_dense_indexer              indexer_type;

    //  implement cursor for morton matrix, somewhere
    //  also, morton indexer?

    typedef morton_dense_key<BitMask>          key_type;
    typedef morton_dense_el_cursor<BitMask>    el_cursor_type; 
    
    typedef dilated_int<std::size_t, BitMask, true>   dilated_row_t;
    typedef dilated_int<std::size_t, ~BitMask, true>  dilated_col_t; 

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

    void init(size_type num_rows, size_type num_cols)
    {
	set_ranges(num_rows, num_cols);
	set_to_zero(*this);
    }

  public:
    // if compile time matrix size allocate memory
    morton_dense() : super_memory( memory_need( dim_type().num_rows(), dim_type().num_cols() ) ) 
    {
	init(dim_type().num_rows(), dim_type().num_cols());
    }

    // only sets dimensions, only for run-time dimensions
    explicit morton_dense(mtl::non_fixed::dimensions d) 
	: super_memory( memory_need( d.num_rows(), d.num_cols() ) ) 
    {
	init(d.num_rows(), d.num_cols());
    }

    // Same with separated row and column number
    morton_dense(size_type num_rows, size_type num_cols) 
	: super_memory( memory_need(num_rows, num_cols) )
    {
	init(num_rows, num_cols);
    }

    // sets dimensions and pointer to external data
    explicit morton_dense(mtl::non_fixed::dimensions d, value_type* a) 
      : super_memory(a) 
    { 
	init(d.num_rows(), d.num_cols());
    }

    // same constructor for compile time matrix size
    // sets dimensions and pointer to external data
    explicit morton_dense(value_type* a) : super_memory(a) 
    { 
	BOOST_ASSERT((dim_type::is_static));
	init(dim_type().num_rows(), dim_type().num_cols());
    }


    void change_dim(size_type num_rows, size_type num_cols)
    {
	if (this->extern_memory && (num_rows != this->num_rows() || num_cols != this->num_cols()))
	    throw "Can't change the size of matrices with external memory";

	set_ranges(num_rows, num_cols);
	this->realloc(memory_need(num_rows, num_cols));
    }

#ifndef MTL_UGLY_MULT_HACK
    self& operator=(const self& src)
    {
	change_dim(src.num_rows(), src.num_cols());
	std::copy(src.elements(), src.elements()+src.used_memory(), this->elements());
	// matrix::copy(src, *this);
	return *this;
    }

    template <typename MatrixSrc>
    self& operator=(const MatrixSrc& src)
    {
	matrix::copy(src, *this);
	return *this;
    }
#endif


    value_type operator() (key_type const& key) const
    {
	return this->data[key.dilated_row.dilated_value() + key.dilated_col.dilated_value()];
    }

    void operator()(key_type const& key, value_type const& value)
    {
	this->data[key.dilated_row.dilated_value() + key.dilated_col.dilated_value()]= value;
    }

    const_access_type operator() (size_type row, size_type col) const
    {
	return this->data[dilated_row_t(row).dilated_value() + dilated_col_t(col).dilated_value()];
    }

    value_type& operator() (size_type row, size_type col)
    {
	return this->data[dilated_row_t(row).dilated_value() + dilated_col_t(col).dilated_value()];
    }

#if 0
    size_type used_memory() const
    {
	return my_used_memory;
    }
#endif

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
    size_type my_used_memory;
    
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
template <typename Elt, unsigned long BitMask, typename Parameters>
bool morton_dense<Elt, BitMask, Parameters>::isRoot(const AhnenIndex& index) const {
    return index.getIndex() == 3;
}

// check if the index is a leaf
template <typename Elt, unsigned long BitMask, typename Parameters>
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
  template <class Elt, unsigned long BitMask, class Parameters>
  struct row<morton_dense<Elt, BitMask, Parameters> >
  {
    typedef mtl::detail::row_in_key<morton_dense<Elt, BitMask, Parameters> > type;
  };

  template <class Elt, unsigned long BitMask, class Parameters>
  struct col<morton_dense<Elt, BitMask, Parameters> >
  {
    typedef mtl::detail::col_in_key<morton_dense<Elt, BitMask, Parameters> > type;
  };

  template <class Elt, unsigned long BitMask, class Parameters>
  struct const_value<morton_dense<Elt, BitMask, Parameters> >
  {
    typedef mtl::detail::matrix_const_value_ref<morton_dense<Elt, BitMask, Parameters> > type;
  };

  template <class Elt, unsigned long BitMask, class Parameters>
  struct value<morton_dense<Elt, BitMask, Parameters> >
  {
    typedef mtl::detail::matrix_value_ref<morton_dense<Elt, BitMask, Parameters> > type;
  };

} // namespace traits


// ================
// Range generators
// ================

namespace traits
{

// ===========
// For cursors
// ===========

    template <class Elt, unsigned long BitMask, class Parameters>
    struct range_generator<glas::tag::all, morton_dense<Elt, BitMask, Parameters> >
    {
	typedef morton_dense<Elt, BitMask, Parameters>        Matrix;
	typedef complexity_classes::linear_cached        complexity;
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

    template <class Elt, unsigned long BitMask, class Parameters>
    struct range_generator<glas::tag::nz, morton_dense<Elt, BitMask, Parameters> >
	: range_generator<glas::tag::all, morton_dense<Elt, BitMask, Parameters> >
    {};

    template <class Elt, unsigned long BitMask, class Parameters>
    struct range_generator<glas::tag::row, morton_dense<Elt, BitMask, Parameters> >
	: detail::all_rows_range_generator<morton_dense<Elt, BitMask, Parameters>, complexity_classes::linear_cached> 
    {};

    // For a cursor pointing to some row give the range of elements in this row 
    template <class Elt, unsigned long BitMask, class Parameters>
    struct range_generator<glas::tag::nz, 
			   detail::sub_matrix_cursor<morton_dense<Elt, BitMask, Parameters>, glas::tag::row, 2> >
    {
	typedef morton_dense<Elt, BitMask, Parameters>                   matrix;
	typedef detail::sub_matrix_cursor<matrix, glas::tag::row, 2>  cursor;
	typedef complexity_classes::linear_cached                        complexity;
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

    template <class Elt, unsigned long BitMask, class Parameters>
    struct range_generator<glas::tag::all, 
			   detail::sub_matrix_cursor<morton_dense<Elt, BitMask, Parameters>, glas::tag::row, 2> >
        : range_generator<glas::tag::nz, 
			  detail::sub_matrix_cursor<morton_dense<Elt, BitMask, Parameters>, glas::tag::row, 2> >
    {};

    template <class Elt, unsigned long BitMask, class Parameters>
    struct range_generator<glas::tag::col, morton_dense<Elt, BitMask, Parameters> >
	: detail::all_cols_range_generator<morton_dense<Elt, BitMask, Parameters>, complexity_classes::linear_cached> 
    {};

    // For a cursor pointing to some row give the range of elements in this row 
    template <class Elt, unsigned long BitMask, class Parameters>
    struct range_generator<glas::tag::nz, 
			   detail::sub_matrix_cursor<morton_dense<Elt, BitMask, Parameters>, glas::tag::col, 2> >
    {
	typedef morton_dense<Elt, BitMask, Parameters>                   matrix;
	typedef detail::sub_matrix_cursor<matrix, glas::tag::col, 2>  cursor;
	typedef complexity_classes::linear_cached                        complexity;
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

    template <class Elt, unsigned long BitMask, class Parameters>
    struct range_generator<glas::tag::all, 
			   detail::sub_matrix_cursor<morton_dense<Elt, BitMask, Parameters>, glas::tag::col, 2> >
        : range_generator<glas::tag::nz, 
			  detail::sub_matrix_cursor<morton_dense<Elt, BitMask, Parameters>, glas::tag::col, 2> >
    {};


// =============
// For iterators
// =============

    namespace detail {

        template <typename OuterTag, typename Matrix, bool is_const>
        struct morton_dense_iterator_range_generator
        {
	    typedef Matrix                                                                matrix_type;
	    typedef typename matrix_type::size_type                                       size_type;
	    typedef typename matrix_type::value_type                                      value_type;
	    typedef typename matrix_type::parameters                                      parameters;
	    typedef detail::sub_matrix_cursor<matrix_type, OuterTag, 2>                   cursor;

	    typedef complexity_classes::linear_cached                                     complexity;
	    static int const                                                              level = 1;

	    typedef typename boost::mpl::if_<
		boost::is_same<OuterTag, glas::tag::row>
	      , typename boost::mpl::if_c<
    	            is_const 
		  , morton_dense_col_const_iterator<Matrix>
		  , morton_dense_col_iterator<Matrix>
		>::type
	      , typename boost::mpl::if_c<
    	            is_const 
		  , morton_dense_row_const_iterator<Matrix>
		  , morton_dense_row_iterator<Matrix>
		>::type
	    >::type type;  

        private:

	    typedef typename boost::mpl::if_c<is_const, const Matrix&, Matrix&>::type    mref_type; 

	    type begin_dispatch(cursor const& c, glas::tag::row)
	    {
		return type(const_cast<mref_type>(c.ref), c.key, c.ref.begin_col());
	    }
	    
	    type end_dispatch(cursor const& c, glas::tag::row)
	    {
		return type(const_cast<mref_type>(c.ref), c.key, c.ref.end_col());
	    }

	    type begin_dispatch(cursor const& c, glas::tag::col)
	    {
		return type(const_cast<mref_type>(c.ref), c.ref.begin_row(), c.key);
	    }

	    type end_dispatch(cursor const& c, glas::tag::col)
	    {
		return type(const_cast<mref_type>(c.ref), c.ref.end_row(), c.key);
	    }

        public:

	    type begin(cursor const& c)
	    {
		return begin_dispatch(c, OuterTag());
	    }

	    type end(cursor const& c)
	    {
		return end_dispatch(c, OuterTag());
	    }	
        };

    } // namespace detail

        
    template <typename Value, unsigned long BitMask, typename Parameters, typename OuterTag>
    struct range_generator<tag::iter::nz, 
			   detail::sub_matrix_cursor<morton_dense<Value, BitMask, Parameters>, OuterTag, 2> >
      : public detail::morton_dense_iterator_range_generator<OuterTag, morton_dense<Value, BitMask, Parameters>, false>
    {};

    template <typename Value, unsigned long BitMask, typename Parameters, typename OuterTag>
    struct range_generator<tag::iter::all, 
			   detail::sub_matrix_cursor<morton_dense<Value, BitMask, Parameters>, OuterTag, 2> >
      : public detail::morton_dense_iterator_range_generator<OuterTag, morton_dense<Value, BitMask, Parameters>, false>
    {};

    template <typename Value, unsigned long BitMask, typename Parameters, typename OuterTag>
    struct range_generator<tag::const_iter::nz, 
			   detail::sub_matrix_cursor<morton_dense<Value, BitMask, Parameters>, OuterTag, 2> >
      : public detail::morton_dense_iterator_range_generator<OuterTag, morton_dense<Value, BitMask, Parameters>, true>
    {};

    template <typename Value, unsigned long BitMask, typename Parameters, typename OuterTag>
    struct range_generator<tag::const_iter::all, 
			   detail::sub_matrix_cursor<morton_dense<Value, BitMask, Parameters>, OuterTag, 2> >
      : public detail::morton_dense_iterator_range_generator<OuterTag, morton_dense<Value, BitMask, Parameters>, true>
    {};


} // namespace traits


// ==========
// Sub matrix
// ==========

template <typename Value, unsigned long BitMask, typename Parameters>
struct sub_matrix_t<morton_dense<Value, BitMask, Parameters> >
{
    typedef morton_dense<Value, BitMask, Parameters>    matrix_type;
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
