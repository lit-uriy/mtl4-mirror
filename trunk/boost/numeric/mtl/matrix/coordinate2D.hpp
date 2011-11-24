
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

namespace mtl {

  namespace matrix {
    
  

//
// Sparse matrix structure in coordinate format
//
// S can be coordinate_sparse_structure<> or coordinate_sparse_structure<> const& or coordinate_sparse_structure<>&
//

template <typename T >
class coordinate2D {
public:
 
  typedef T                                                                 value_type ;
  typedef T&                                                                reference ;
  typedef T const&                                                          const_reference ;
  typedef unsigned int				                            size_type ;

  typedef mtl::dense_vector< size_type >                                          row_index_array_type ;
  typedef mtl::dense_vector< size_type >                                          column_index_array_type ;
  typedef mtl::dense_vector< value_type >                                         value_array_type ;

public:
  explicit coordinate2D( size_type rows, size_type cols, size_type nnz )
  : row_( nnz), col_(nnz), values_( nnz ), nrows(rows), ncols(cols), nnz_( nnz ), counter(0)
  {
    row_= 0;
    col_= 0;
    values_= 0;
    
     
  } 
  
  explicit coordinate2D( size_type rows, size_type cols )
  : row_( 0), col_(0), values_( 0 ), nrows(rows), ncols(cols), nnz_( 0 ), counter(0)
  {
     
  }
  //copy constructor
 explicit coordinate2D( coordinate2D const& that )
  : row_( that.row_), col_(that.col_), values_( that.values_ ), nrows(that.nrows), ncols(that.ncols), nnz_(that.nnz_)
  {}


public:
  size_type num_rows() const {
    return nrows ;
  }
  
  size_type num_cols() const {
    return ncols ;
  }

  size_type const& nnz() const {
    return nnz_ ; 
  }

  value_array_type const& value_array() const {
    return values_ ;
  }

  value_array_type& value_array() {
    return values_ ;
  }

  row_index_array_type const& row_index_array() const {
    return row_ ;
  }

  column_index_array_type const& column_index_array() const {
    return col_ ;
  }
  
  row_index_array_type& row_index_array() {
    return row_ ;
  }

  column_index_array_type& column_index_array() {
    return col_ ;
  }
public:
  ///insert an entry at the end of the row-,col- and value-array  expensive in mtl4
  void push_back( size_type r, size_type c, const_reference v ) {
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
  
  /// overrides an existing entry, if pos < nnz and (r,c) in (rows,cols)
  void insert( size_type r, size_type c, const_reference v, size_type pos ) {
    assert(!is_negative(r)  || !(r >= this->num_rows()) ); //TODO check if we need this really
    assert(!is_negative(c) || (c >= this->num_cols()) ); //TODO check if we need this really
    assert(!is_negative(pos) || (pos >= this->nnz_) ); //TODO check if we need this really
 
     row_[pos]= r;
     col_[pos]= c;
     values_[pos] = v;
     counter++;
  }
  
  
  
  /// compress matrix, cut 0
  void compress(){
 if(counter>0){
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
    }else{
      row_.change_dim(0);   
      col_.change_dim(0);   
      values_.change_dim(0); 
      nnz_=0;
    }

  }
  
  /// sorting standard by row
  void sort(){
    sort_row();
  }
  ///sorting by rows
  void sort_row(){
    if(size(this->value_array())>0){
      mtl::vector::sort(row_,col_,values_);
    }
  }
  ///sorting by columns
  void sort_col(){
    if(size(this->value_array())>0){
      mtl::vector::sort(col_,row_,values_);
    }
  }
  
   void print(){
      std::cout<<"row=" << row_ << "\n";
      std::cout<<"col=" << col_ << "\n";
      std::cout<<"val=" << values_ << "\n";
      
   }
  ///operator * for  vector= coordinaten-matrix * vector
  template <typename Vector >
  Vector operator*(const Vector& x){
    assert(ncols == size(x));
    Vector res(nrows);
    res= 0;
    for(size_type i= 0; i < size(values_); i++){
      res[row_[i]]=  res[row_[i]] + x[col_[i]]*values_[i];
    }
    return res;
  }
  
   


#if 1
public:
  value_type operator() (const size_type r,const  size_type c ) {
    assert(is_negative(r) || !(r >= this->num_rows()) ); //TODO check if we need this really
    assert(is_negative(c) || !(c >= this->num_cols()) ); //TODO check if we need this really
      value_type zero(0.0);
      size_type i(0);
    for(; i < nnz(); i++){
      if(row_[i] == r && col_[i] == c){
	break;
      } else if(row_[i] > r && col_[i] > c){
	break;
      }
    }
     return i<=nnz()-1 ? values_[i] :zero;
  }
  

#endif
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
mtl::matrix::compressed2D<Value> crs(coordinate2D<Value> const& that){
      typedef typename coordinate2D<Value>::size_type  size_type;
      size_type row=num_rows(that), col=num_cols(that), nz=nnz(that);
      
      mtl::matrix::compressed2D<Value> C(row,col);
      {
	mtl::matrix::inserter<mtl::compressed2D<Value> > ins(C);
	for(size_type i= 0; i < nz; i++){
	  ins[that.row_index_array()[i]][that.column_index_array()[i]] << that.value_array()[i];
	}
      }
      return C;
}


} // namespace matrix
} // namespace mtl

#endif // MTL_COORDINATE2D_INCLUDE

