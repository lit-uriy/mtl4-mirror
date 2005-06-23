// $COPYRIGHT$

#ifndef MTL_DENSE_MATRIX_INCLUDE
#define MTL_DENSE_MATRIX_INCLUDE



namespace mtl {

  /*
template <class ELT, class Offset>
class dense_cursor {
public:
  typedef typename Offset::size_type size_type;
  typedef std::pair<size_type,size_type> pair_type;

  dense_cursor () {} 
  dense_cursor (

  */

template <class ELT, // class OffsetGen, 
	  int MM = 0, int NN = 0>
class dense_matrix : detail::base_matrix<ELT> {
  typedef detail::base_matrix<ELT>      basem;
public:	
  typedef ELT                           value_type;
  
}; // dense_matrix

} /* namespace mtl */

#endif /* MTL_DENSE_MATRIX_INCLUDE */
