#include <iostream>

#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/dimensions.hpp>
#include <boost/numeric/mtl/dense2D.hpp>

using namespace mtl;

template <typename Matrix, bool Enable>
struct array_size
{
  // More convenient when always exist (and then not use it)
  static std::size_t const value= 0;
};

#if 0
template <typename Elt, typename Orientation, typename Index, std::size_t Rows, std::size_t Cols, bool OnHeap>
struct array_size<mtl::dense2d<Elt, mtl::matrix_parameters<Orientation, Index, 
							    mtl::fixed::dimensions<Rows, Cols>, OnHeap> > >
{  
  static unsigned const value= Rows * Cols;
};
#endif

template <typename Elt, typename Parameters>
struct array_size<mtl::dense2D<Elt, Parameters>, true>
{
  typedef typename Parameters::dimensions   dimensions;
  BOOST_STATIC_ASSERT((dimensions::is_static));
  static std::size_t const value= dimensions::Num_Rows * dimensions::Num_Cols;
};

template <typename Elt, bool OnStack, unsigned Size= 0>
struct generic_array
{
  Elt    *data;
};

template <typename Elt, unsigned Size>
struct generic_array<Elt, true, Size>
{
  Elt    data[Size];
};

#if 0
template <typename Matrix>
struct contiguous_memory_matrix
  : public generic_array<Matrix::value_type, Matrix::on_stack, 
			 array_size<Matrix, Matrix::on_stack>::value >
{
  // typedef typename Matrix::value_type    value_type;
};

#endif 

struct s1 : public generic_array<float, false>
{
  int a;
};

struct s2 : public generic_array<int, true, 10>
{
  float b;
};

template <typename Matrix>
struct basis
  : public generic_array<Matrix::value_type, true, 10>
{};

template <typename Elt>
struct superclass : public basis<superclass<Elt> >
{
  typedef Elt    value_type;
};



int main(int argc, char* argv[])
{

  superclass<double> sc;

  typedef matrix_parameters<row_major, mtl::index::c_index, fixed::dimensions<2, 3> >         parameters1;
  typedef matrix_parameters<row_major, mtl::index::c_index, fixed::dimensions<2, 3>, true >   parameters2;

#if 0
  contiguous_memory_matrix< mtl::dense2D<float, parameters1> > m1;
  m1.data= new  float[6];
  m1.data[3]= 11;

  contiguous_memory_matrix< mtl::dense2D<float, parameters2> > m2;
  m2.data[3]= 12;
#endif

  s1 ss1;
  s2 ss2;
  
  ss1.a = 2; 
  ss2.b = 3.0; 

  std::cout << ss1.a << ss2.b << '\n';

    return 0;
}
