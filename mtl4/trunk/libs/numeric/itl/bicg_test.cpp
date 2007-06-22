#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/bicgstab.hpp>
#include <boost/numeric/itl/itl.hpp>

using namespace mtl;
using namespace itl;

int main()
{
  const int size = 1000, N = size * size;

  compressed2D<double> A(N, N);
  matrix::laplacian_setup(A, size, size);

  dense_vector<double> x(N, 1.0), b(N);

  b = A * x;
  x= 0;

  noisy_iteration<double> iter(b, N, 1.e-6);
  bicgstab(A, x, b, iter);

  return 0;
}
