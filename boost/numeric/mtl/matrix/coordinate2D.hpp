
#ifndef MTL_COORDINATE2D_INCLUDE
#define MTL_COORDINATE2D_INCLUDE



#include <boost/tuple/tuple.hpp>
#include <vector>
#include <cassert>
#include <boost/numeric/mtl/operation/is_negative.hpp>
#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/operation/sort.hpp>
#include <boost/numeric/mtl/operation/iota.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>

namespace mtl {  namespace matrix {
    
  
/// Sparse matrix structure in coordinate format
/**
	S can be coordinate_sparse_structure<> or coordinate_sparse_structure<> const& or coordinate_sparse_structure<>&
 **/
template <typename T >
class coordinate2D 
{
  public:
 
    typedef T                                                                 value_type ;
    typedef T&                                                                reference ;
    typedef T const&                                                          const_reference ;
    typedef unsigned int				                            size_type ;

    typedef mtl::dense_vector< size_type >                                    row_index_array_type ;
    typedef mtl::dense_vector< size_type >                                    column_index_array_type ;
    typedef mtl::dense_vector< value_type >                                   value_array_type ;

    /// Common constructor
    explicit coordinate2D( size_type rows, size_type cols, size_type nnz= 0 )
      : row_( nnz, size_type(0)), col_(nnz, size_type(0)), values_( nnz, value_type(0) ), 
	nrows(rows), ncols(cols), nnz_( nnz ), counter(0)
    {} 
  
    /// Copy constructor
    explicit coordinate2D( coordinate2D const& that )
      : row_( that.row_), col_(that.col_), values_( that.values_ ), nrows(that.nrows), 
	ncols(that.ncols), nnz_(that.nnz_)
    {}

    size_type num_rows() const { return nrows ;  }  
    size_type num_cols() const { return ncols ;  }
    size_type const& nnz() const { return nnz_ ; } ///< Number of non-zeros

    value_array_type const& value_array() const { return values_ ; } ///< Array of values (const)
    value_array_type& value_array() { return values_ ; } ///< Array of values (mutable)

    row_index_array_type const& row_index_array() const { return row_ ; } ///< Array of rows  (const)
    column_index_array_type const& column_index_array() const {	return col_ ; } ///< Array of columns (const)
  
    row_index_array_type& row_index_array() { return row_ ; } ///< Array of rows   (mutable)
    column_index_array_type& column_index_array() { return col_ ; } ///< Array of columns  (mutable)

    /// Insert an entry at the end of the row-,col- and value-array  expensive in mtl4
    void push_back( size_type r, size_type c, const_reference v ) 
    {
	row_index_array_type      tmp_r(row_);
	column_index_array_type   tmp_c(col_);
	value_array_type          tmp_v(values_);
	nnz_++;
	row_.change_dim(nnz_);
	col_.change_dim(nnz_);
	values_.change_dim(nnz_);
	if(nnz_> 1){
	    irange ra(0,nnz_-1);
	    row_[ra]= tmp_r;
	    col_[ra]= tmp_c;
	    values_[ra]= tmp_v;
	}
	row_[nnz_-1]= r;
	col_[nnz_-1]= c;
	values_[nnz_-1] = v;
    } // push_back
  
    /// Overrides an existing entry, if pos < nnz and (r,c) in (rows,cols)
    void insert( size_type r, size_type c, const_reference v, size_type pos ) 
    {
	assert(!is_negative(r)  || !(r >= this->num_rows()) ); //TODO check if we need this really
	assert(!is_negative(c) || (c >= this->num_cols()) ); //TODO check if we need this really
	assert(!is_negative(pos) || (pos >= this->nnz_) ); //TODO check if we need this really
 
	row_[pos]= r;
	col_[pos]= c;
	values_[pos] = v;
	counter++;
    }
  
    /// Compress matrix, cut 0
    void compress()
    {
	if(counter>0) {
	    row_index_array_type      rowi;
	    column_index_array_type   coli;
	    value_array_type          valuesi ;
	    rowi=     row_[irange(0,counter)];
	    coli=     col_[irange(0,counter)];
	    valuesi=  values_[irange(0,counter)];
	    row_.change_dim(counter);
	    col_.change_dim(counter);
	    values_.change_dim(counter);
	    row_= rowi;
	    col_= coli;
	    values_= valuesi;
	    nnz_= counter;
	} else {
	    row_.change_dim(0);   
	    col_.change_dim(0);   
	    values_.change_dim(0); 
	    nnz_=0;
	}

    }
  
    /// sorting standard by row
    void sort(){ sort_row();  }

    ///sorting by rows
    void sort_row()
    {
	if(size(this->value_array())>0)
	    mtl::vector::sort(row_, col_, values_);
    }

    ///sorting by columns
    void sort_col()
    {
	if(size(this->value_array())>0)
	    mtl::vector::sort(col_, row_, values_);
    }
  
    void print()
    {
	std::cout<< "row=" << row_ << "\n";
	std::cout<< "col=" << col_ << "\n";
	std::cout<< "val=" << values_ << "\n";
      
    }

    ///operator * for  vector= coordinaten-matrix * vector
    template <typename Vector >
    Vector operator*(const Vector& x)
    {
	assert(ncols == size(x));
	Vector res(nrows);
	res= 0;
	for(size_type i= 0; i < size(values_); i++)
	    res[row_[i]]=  res[row_[i]] + x[col_[i]]*values_[i];
	return res;
    }
  
    value_type operator() (const size_type r,const  size_type c ) 
    {
	assert(is_negative(r) || !(r >= this->num_rows()) ); //TODO check if we need this really
	assert(is_negative(c) || !(c >= this->num_cols()) ); //TODO check if we need this really
	value_type zero(0.0);
	for(size_type i= 0; i < nnz(); i++) 
	    if(row_[i] == r && col_[i] == c)
		return values_[i];
	    else if(row_[i] > r && col_[i] > c)
		return zero;
	return zero;
    }
  

  private:
    row_index_array_type      row_;
    column_index_array_type   col_;
    value_array_type          values_ ;
    size_type                 nrows, ncols, nnz_, counter;
};

// ================
// Free functions
// ================


/// Number of rows
template <typename Value>
typename coordinate2D<Value>::size_type
inline num_rows(const coordinate2D<Value>& matrix)
{
    return matrix.num_rows();
}

/// Number of columns
template <typename Value>
typename coordinate2D<Value>::size_type
inline num_cols(const coordinate2D<Value>& matrix)
{
    return matrix.num_cols();
}

/// Size of the matrix, i.e. the number of row times columns
template <typename Value>
typename coordinate2D<Value>::size_type
inline size(const coordinate2D<Value>& matrix)
{
    return matrix.num_cols() * matrix.num_rows();
}

/// Number of NoZeros of the matrix
template <typename Value>
typename coordinate2D<Value>::size_type
inline nnz(const coordinate2D<Value>& matrix)
{
    return matrix.nnz();
}

///returns a matrix in crs format from a coordinateformat
template <typename Value>
mtl::matrix::compressed2D<Value> crs(coordinate2D<Value> const& that)
{
    typedef typename coordinate2D<Value>::size_type  size_type;
    size_type row= num_rows(that), col= num_cols(that), nz= nnz(that);
      
    mtl::matrix::compressed2D<Value> C(row,col);
    {
	mtl::matrix::inserter<mtl::compressed2D<Value> > ins(C);
	for(size_type i= 0; i < nz; i++)
	    ins[that.row_index_array()[i]][that.column_index_array()[i]] << that.value_array()[i];
     
    }
    return C;
}



template <typename Matrix, typename Updater = mtl::operations::update_store<typename Matrix::value_type> >
struct coordinate2D_inserter
{
    BOOST_STATIC_ASSERT((boost::is_same<Updater, mtl::operations::update_store<typename Matrix::value_type> >::value));

    typedef coordinate2D_inserter                       self;
    typedef Matrix                                      matrix_type;
    typedef typename matrix_type::size_type             size_type;
    typedef typename matrix_type::value_type            value_type;
    typedef operations::update_proxy<self, size_type>   proxy_type;
    
    // We only support storing so far !!!
    BOOST_STATIC_ASSERT((boost::is_same<Updater, mtl::operations::update_store<value_type> >::value));

    explicit coordinate2D_inserter(matrix_type& matrix, size_type) : matrix(matrix) {}

    
 private:

    struct update_proxy
    {
	// self is type of inserter not update_proxy !!!
	update_proxy(self& ref, size_type row, size_type col) : ref(ref), row(row), col(col) {}

	template <typename Value>
	update_proxy& operator<< (Value const& val)
	{
	    ref.matrix.push_back(row, col, val);
	    return *this;
	}
	self& ref;
	size_type row, col;
    };
    
    proxy_type operator() (size_type row, size_type col)
    {
	return proxy_type(*this, row, col);
    }

    
    struct bracket_proxy
    {
	bracket_proxy(self& ref, size_type row) : ref(ref), row(row) {}
	
	proxy_type operator[](size_type col)
	{
	    return proxy_type(ref, row, col);
	}

	self&      ref;
	size_type  row;
    };

  public:

    bracket_proxy operator[] (size_type row)
    {
	return bracket_proxy(*this, row);
    }

#if 0
    template <typename Value>
    void update(size_type row, size_type col, Value val)
    {
	Updater() (matrix(row, col), val);
    }

    template <typename Modifier, typename Value>
    void modify(size_type row, size_type col, Value val)
    {
	Modifier() (matrix(row, col), val);
    }

    template <typename EMatrix, typename Rows, typename Cols>
    self& operator<< (const matrix::element_matrix_t<EMatrix, Rows, Cols>& elements)
    {
	using mtl::size;
	for (unsigned ri= 0; ri < size(elements.rows); ri++)
	    for (unsigned ci= 0; ci < size(elements.cols); ci++)
		update (elements.rows[ri], elements.cols[ci], elements.matrix(ri, ci));
	return *this;
    }

    template <typename EMatrix, typename Rows, typename Cols>
    self& operator<< (const matrix::element_array_t<EMatrix, Rows, Cols>& elements)
    {
	using mtl::size;
	for (unsigned ri= 0; ri < size(elements.rows); ri++)
	    for (unsigned ci= 0; ci < size(elements.cols); ci++)
		update (elements.rows[ri], elements.cols[ci], elements.array[ri][ci]);
	return *this;
    }
#endif

  protected:
    matrix_type&         matrix;
};



} // namespace matrix
} // namespace mtl

#endif // MTL_COORDINATE2D_INCLUDE

