// $COPYRIGHT$

#ifndef MTL_MORTON_DENSE_INCLUDE
#define MTL_MORTON_DENSE_INCLUDE

#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/detail/base_cursor.hpp>
#include <boost/numeric/mtl/detail/base_matrix.hpp>
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
struct morton_dense_el_cursor
{
    typedef std::size_t    size_t;
    typedef dilated_int<std::size_t, BitMask, true>   dilated_row_t;
    typedef dilated_int<std::size_t, ~BitMask, true>  dilated_col_t; 
    typedef morton_dense_el_cursor                    self;

    morton_dense_el_cursor(size_t my_row, size_t my_col, size_t num_cols) 
	: my_row(my_row), my_col(my_col), dilated_row(my_row), dilated_col(my_col), num_cols(num_cols) 
    {}

    self& operator++ ()
    {
	++my_col; ++dilated_col;
	if (my_col == num_cols) {
	    my_col= 0; dilated_col= dilated_col_t(0);
	    ++my_row; ++dilated_row;
	}
	return *this;
    }

    self& operator* ()
    {
	return *this;
    }

    bool operator== (self const& x)
    {
	assert (num_cols == x.num_cols);
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

protected:
    size_t                       my_row, my_col;   
    dilated_row_t                dilated_row;
    dilated_col_t                dilated_col; 
    size_t                       num_cols;
};


// Morton Dense matrix type
template <typename Elt, typename Parameters>
class morton_dense : public detail::base_matrix<Elt, Parameters>, 
		     public detail::contiguous_memory_matrix<Elt, false>
{
    typedef detail::base_matrix<Elt, Parameters>                super;
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

    typedef morton_dense_el_cursor<0x55555555>    key_type;
    typedef std::size_t                           size_type;
    
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



  public:
    // if compile time matrix size allocate memory
    morton_dense() : super(), super_memory( memory_need( dim_type().num_rows(), dim_type().num_cols() ) ) {}

    // only sets dimensions, only for run-time dimensions
    explicit morton_dense(mtl::non_fixed::dimensions d) 
	: super(d), super_memory( memory_need( d.num_rows(), d.num_cols() ) ) 
    {
	set_nnz();
    }

    // sets dimensions and pointer to external data
    explicit morton_dense(mtl::non_fixed::dimensions d, value_type* a) 
      : super(d), super_memory(a) 
    { 
        set_nnz();
    }

    // same constructor for compile time matrix size
    // sets dimensions and pointer to external data
    explicit morton_dense(value_type* a) : super(), super_memory(a) 
    { 
	BOOST_ASSERT((dim_type::is_static));
    }


  protected:
    void set_nnz()
    {
      this->nnz = this->dim.num_rows() * this->dim.num_cols();
    }
    
    size_type memory_need(size_type rows, size_type cols)
    {
	return 3; // change this
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
template <typename Elt, typename Parameters>
bool morton_dense<Elt, Parameters>::isRoot(const AhnenIndex& index) const {
    return index.getIndex() == 3;
}

// check if the index is a leaf
template <typename Elt, typename Parameters>
bool morton_dense<Elt, Parameters>::isLeaf(const AhnenIndex& index) const {
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
  template <class Elt, class Parameters>
  struct row<morton_dense<Elt, Parameters> >
  {
    typedef mtl::detail::row_in_key<morton_dense<Elt, Parameters> > type;
  };

  template <class Elt, class Parameters>
  struct col<morton_dense<Elt, Parameters> >
  {
    typedef mtl::detail::col_in_key<morton_dense<Elt, Parameters> > type;
  };

  template <class Elt, class Parameters>
  struct const_value<morton_dense<Elt, Parameters> >
  {
    typedef mtl::detail::direct_const_value<morton_dense<Elt, Parameters> > type;
  };

  template <class Elt, class Parameters>
  struct value<morton_dense<Elt, Parameters> >
  {
    typedef mtl::detail::direct_value<morton_dense<Elt, Parameters> > type;
  };

  template <class Elt, class Parameters>
  struct is_mtl_type<morton_dense<Elt, Parameters> >
  {
    static bool const value= true;
  };

  // define corresponding type without all template parameters
  template <class Elt, class Parameters>
  struct matrix_category<morton_dense<Elt, Parameters> >
  {
    typedef mtl::tag::morton_dense type;
  };

} // namespace traits


template <class Elt, class Parameters>
inline typename traits::row<morton_dense<Elt, Parameters> >::type
row(const morton_dense<Elt, Parameters>& ma)
{
  return typename traits::row<morton_dense<Elt, Parameters> >::type(ma);
}

template <class Elt, class Parameters>
inline typename traits::col<morton_dense<Elt, Parameters> >::type
col(const morton_dense<Elt, Parameters>& ma)
{
  return typename traits::col<morton_dense<Elt, Parameters> >::type(ma);
}

template <class Elt, class Parameters>
inline typename traits::const_value<morton_dense<Elt, Parameters> >::type
const_value(const morton_dense<Elt, Parameters>& ma)
{
  return typename traits::const_value<morton_dense<Elt, Parameters> >::type(ma);
}

template <class Elt, class Parameters>
inline typename traits::value<morton_dense<Elt, Parameters> >::type
value(const morton_dense<Elt, Parameters>& ma)
{
  return typename traits::value<morton_dense<Elt, Parameters> >::type(ma);
}


// Range generators
// ================


namespace traits
{

#if 0

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

      // what's the complexity of morton cursor?
      template <typename Orientation> struct morton_dense_rc {};
        template<> struct morton_dense_rc<row_major>
        {
	  typedef complexity::linear type;
        };

        template<> struct morton_dense_rc<col_major>
        {
	  typedef complexity::linear type;
        };

        template<> struct morton_dense_rc<morton_major>
        {
	  typedef complexity::linear_cached type;
        };


      // Complexity of column cursor is of course opposite
        template <typename Orientation> struct morton_dense_cc
	: morton_dense_rc<typename transposed_orientation<Orientation>::type>
        {};
    }


  template <class Elt, class Parameters>
  struct range_generator<glas::tags::row_t, morton_dense<Elt, Parameters> >
    : detail::all_rows_range_generator<morton_dense<Elt, Parameters>,
				       typename detail::morton_dense_rc<typename Parameters::orientation>::type>
  {};



  // For a cursor pointing to some row give the range of elements in this row
  template <class Elt, class Parameters>
  struct range_generator<glas::tags::nz_t,
			 detail::sub_matrix_cursor<morton_dense<Elt, Parameters>, glas::tags::row_t, 2> >
  {
    typedef morton_dense<Elt, Parameters>  matrix;
    typedef detail::sub_matrix_cursor<matrix, glas::tags::row_t, 2> cursor;

    // linear for col_major and linear_cached for row_major
    typedef typename detail::morton_dense_rc<typename Parameters::orientation>::type   complexity;

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
			 detail::sub_matrix_cursor<morton_dense<Elt, Parameters>, glas::tags::row_t, 2> >
  : range_generator<glas::tags::nz_t,
		    detail::sub_matrix_cursor<morton_dense<Elt, Parameters>, glas::tags::row_t, 2> >
  {};

  template <class Elt, class Parameters>
  struct range_generator<glas::tags::col_t, morton_dense<Elt, Parameters> >
    : detail::all_cols_range_generator<morton_dense<Elt, Parameters>,
				       typename detail::morton_dense_cc<typename Parameters::orientation>::type>
  {};


  // For a cursor pointing to some row give the range of elements in this row
  template <class Elt, class Parameters>
  struct range_generator<glas::tags::nz_t,
			 detail::sub_matrix_cursor<morton_dense<Elt, Parameters>, glas::tags::col_t, 2> >
  {
    typedef morton_dense<Elt, Parameters>  matrix;
    typedef detail::sub_matrix_cursor<matrix, glas::tags::col_t, 2> cursor;
    typedef typename detail::morton_dense_cc<typename Parameters::orientation>::type   complexity;
    static int const             level = 1;
    /*
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
    */
  };

  template <class Elt, class Parameters>
  struct range_generator<glas::tags::all_t,
			 detail::sub_matrix_cursor<morton_dense<Elt, Parameters>, glas::tags::col_t, 2> >
  : range_generator<glas::tags::nz_t,
		    detail::sub_matrix_cursor<morton_dense<Elt, Parameters>, glas::tags::col_t, 2> >
  {};
#endif

} // namespace traits



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



} // namespace mtl

#endif // MTL_MORTON_DENSE_INCLUDE
