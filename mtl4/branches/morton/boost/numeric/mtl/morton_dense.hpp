// $COPYRIGHT$

#ifndef MTL_MORTON_DENSE_INCLUDE
#define MTL_MORTON_DENSE_INCLUDE

#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/detail/base_cursor.hpp>
#include <boost/numeric/mtl/detail/base_sub_matrix.hpp>
#include <boost/numeric/mtl/detail/contiguous_memory_matrix.hpp>
#include <boost/numeric/mtl/detail/dilated_int.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/detail/range_generator.hpp>
#include <boost/numeric/mtl/property_map.hpp>
#include <boost/numeric/mtl/range_generator.hpp>
#include <boost/numeric/mtl/detail/range_generator.hpp>
#include <boost/numeric/mtl/complexity.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>
#include <boost/numeric/mtl/ahnentafel_index.hpp>
#include <vector>

namespace mtl {

template <std::size_t  BitMask>
struct morton_dense_key
{
    typedef std::size_t                               size_t;
    typedef dilated_int<std::size_t, BitMask, true>   dilated_row_t;
    typedef dilated_int<std::size_t, ~BitMask, true>  dilated_col_t; 
    typedef morton_dense_key                          self;

    morton_dense_key(size_t my_row, size_t my_col) 
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

    size_t row() const
    {
	return my_row;
    }

    size_t col() const
    {
	return my_col;
    }

public:
    size_t                       my_row, my_col;   
    dilated_row_t                dilated_row;
    dilated_col_t                dilated_col; 
};

template <std::size_t  BitMask>
struct morton_dense_el_cursor 
    : public morton_dense_key<BitMask>
{
    typedef std::size_t                               size_t;
    typedef dilated_int<std::size_t, ~BitMask, true>  dilated_col_t; 
    typedef morton_dense_el_cursor                    self;
    typedef morton_dense_key<BitMask>                 base;

    morton_dense_el_cursor(size_t my_row, size_t my_col, size_t num_cols) 
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
    // typedef morton_dense_el_cursor<Elt>       el_cursor_type;  
    // typedef morton_dense_indexer              indexer_type;

    //  implement cursor for morton matrix, somewhere
    //  also, morton indexer?

    typedef morton_dense_key<0x55555555>          key_type;
    typedef morton_dense_el_cursor<0x55555555>    el_cursor_type; 
    
    typedef dilated_int<std::size_t, BitMask, true>   dilated_row_t;
    typedef dilated_int<std::size_t, ~BitMask, true>  dilated_col_t; 
    //  typedef morton_dense_indexer              indexer_type;


  public: 

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

  protected:
    
    // ranges of rows and columns
    dilated_row_t            my_begin_row, my_end_row;
    dilated_col_t            my_begin_col, my_end_col;

    void set_ranges(size_type er, size_type ec)
    {
	super::set_ranges(0, er, 0, ec);
	my_begin_row= 0; my_end_row= er;
	my_begin_col= 0; my_end_col= ec;
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
      this->nnz = this->dim.num_rows() * this->dim.num_cols();
    }
    
    size_type memory_need(size_type rows, size_type cols)
    {
        dilated_row_t n_rows(rows - 1);
        dilated_col_t n_cols(cols - 1);
        return (n_rows.dilated_value() + n_cols.dilated_value() + 1);
    }
    

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
};



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


} // namespace mtl

#endif // MTL_MORTON_DENSE_INCLUDE
