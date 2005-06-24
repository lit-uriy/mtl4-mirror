// $COPYRIGHT$

#ifndef MTL_DENSE_MATRIX_INCLUDE
#define MTL_DENSE_MATRIX_INCLUDE



namespace mtl {

 
template <class ELT> //, class Offset>
class dense_element_cursor : public detail::base_matrix_cursor<const ELT*> {
public:
  typedef ELT                           value_type;
  typedef const value_type*             pointer_type;
  typedef typename Offset::size_type    size_type;
  // typedef std::pair<size_type,size_type> pair_type;

  dense_cursor () {} 
  dense_cursor (pointer_type mme, pointer_type b, int mm, int nn) 
    : 
 
protected:

  pointer_type      mbegin; // start address of matrix data
};


  
  // M and N as template parameters might be considered later
template <class ELT>
class dense_matrix : public detail::base_matrix<ELT> {
  typedef detail::base_matrix<ELT>      basem;
public:	
  typedef ELT                           value_type;
  // typedef const value_type*             pointer_type;
  typedef dense_element_cursor<ELT>     element_cursor_type;
  
  element_cursor_type ebegin() const {
    return element_cursor_type (my_data_const, my_data_const); }
  element_cursor_type eend() const {
    return element_cursor_type (my_data_const+nnz, my_data_const); }

protected:

}; // dense_matrix

} /* namespace mtl */

#endif /* MTL_DENSE_MATRIX_INCLUDE */
