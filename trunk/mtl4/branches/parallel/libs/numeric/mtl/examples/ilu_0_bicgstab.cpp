#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

using namespace mtl;
using namespace itl;

int main(int argc, char* argv[])
{
    const int size = 10, N = size * size; 
    typedef compressed2D<double>  matrix_type;

    // Set up a matrix 100 x 100 with 5-point-stencil
    matrix_type                   A(N, N);
    matrix::laplacian_setup(A, size, size);

    // Create an ILU(0) preconditioner
    pc::ilu_0<matrix_type>        P(A);
    
    // Set b such that x == 1 is solution; start with x == 0
    dense_vector<double>          x(N, 1.0), b(N);
    b= A * x; x= 0;
    
    // Termination criterion: r < 1e-6 * b or N iterations
    noisy_iteration<double>       iter(b, N, 1.e-6);
    
    // Solve Ax == b with left preconditioner P
    bicgstab(A, x, b, P, iter);

    return 0;
}
