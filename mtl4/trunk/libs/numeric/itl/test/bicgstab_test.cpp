#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

using namespace mtl;
using namespace itl;

int main()
{
  // For a more realistic example set size to 1000 or larger
  const int size = 100, N = size * size;

  typedef compressed2D<double>  matrix_type;
  compressed2D<double>          A(N, N);
  matrix::laplacian_setup(A, size, size);

  pc::diagonal<matrix_type>     P(A);
  dense_vector<double>          x(N, 1.0), b(N);

  b = A * x;
  x= 0;

  noisy_iteration<double> iter(b, 500, 1.e-6);
  bicgstab(A, x, b, P, iter);

  return 0;
}
