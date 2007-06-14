#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[])
{
    using namespace mtl;
    
    const unsigned n= 100;
    dense2D<double>                            a(n, n), b(n, n);
    morton_dense<double, doppled_64_row_mask>  c(n, n);

    a= b * b;   // use BLAS
    a= b * c;   // use recursion + tiling from MTL4

    return 0;
}
