#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[])
{
    using namespace mtl;
    
    const unsigned n= 100;
    dense2D<double>                            A(n, n), B(n, n);
    morton_dense<double, doppled_64_row_mask>  C(n, n);
    matrix::hessian_setup(A, 3.0); matrix::hessian_setup(B, 1.0); matrix::hessian_setup(C, 2.0);

    A= B * B + C * B - B * C;   

    return 0;
}
