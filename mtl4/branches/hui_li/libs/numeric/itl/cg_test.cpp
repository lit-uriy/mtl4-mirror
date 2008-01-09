#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/cg.hpp>
#include <boost/numeric/itl/itl.hpp>

using namespace mtl;
using namespace itl;

int main()
{
  // For a more realistic example set size to 1000 or larger
  const int size = 100, N = size * size;

  compressed2D<double> A(N, N);
  matrix::laplacian_setup(A, size, size);

  dense_vector<double> x(N, 1.0), b(N);

  b = A * x;
  x= 0;

  noisy_iteration<double> iter(b, N, 1.e-6);
  cg(A, x, b, iter);

  return 0;
}
