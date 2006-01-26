#include <iostream>

#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/dimensions.hpp>


template <typename Matrix> // , typename Parameters>
struct memory_size
{
  // must be contained in all specializations !!!
  // static unsigned const value;
};

#if 0
template <typename Elt, typename Orientation, typename Index, std::size_t Rows, std::size_t Cols, bool OnHeap>
struct memory_size<mtl::dense2d<Elt, mtl::matrix_parameters<Orientation, Index, 
							    mtl::fixed::dimensions<Rows, Cols>, OnHeap> > >
{  
  static unsigned const value= Rows * Cols;
}
#endif



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

struct s1 : public generic_array<float, false>
{
  int a;
};

struct s2 : public generic_array<int, true, 10>
{
  float b;
};


int main(int argc, char* argv[])
{
  s1 ss1;
  s2 ss2;
  
  ss1.a = 2; 
  ss2.b = 3.0; 

  std::cout << ss1.a << ss2.b << '\n';

    return 0;
}
