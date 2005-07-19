#include <iostream>
#include <vector>
#include <mtl/base_types.hpp>
#include <mtl/dense2D.hpp>
#include <mtl/property_map.hpp>
#include <mtl/mat_vec_mult.hpp>
#include <boost/timer.hpp>

#include <boost/mpl/if.hpp>

int main (int argc, char** argv) {
  using namespace std;
  using namespace mtl;

  if (argc < 2) {
    cout << "syntax: mat_vec_mult_timing size\n"; exit(1); }

  typedef dense2D<double, col_major, c_index> matrix_type; 
  std::size_t     size(atoi(argv[1]));
  matrix_type     matrix(dim_type(size, size), 1);
  std::vector<double>  vin(size, 1), vout(size, 7);
  
  cout << "vin is mtl type: " << is_mtl_type<std::vector<int> >::value
       << " is fortran indexed: " <<  is_fortran_indexed<std::vector<int> >::value << endl;
  cout << "matrix is mtl type: " << is_mtl_type<matrix_type>::value
       << " is fortran indexed: " <<  is_fortran_indexed<matrix_type>::value << endl;

  // cout << "matrix is boost::is_same<typename indexing<T>::type, c_index>::value

  boost::timer ti;
  mat_vec_mult(matrix, vin, vout);
  cout << ti.elapsed() << " s\n";

  for (size_t i= 0; i < size; i++) 
    if (vout[i] != (int) size) cout << "vout[" << i << "] is " << vout[i] << endl;
  
  return 0;
}
