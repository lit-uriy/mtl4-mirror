#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[])
{
    using namespace mtl; using namespace mtl::matrix;
    
    const unsigned                xd= 2, yd= 5, n= xd * yd;
    dense2D<double>               A(n, n);
    laplacian_setup(A, xd, yd); 
    dense_vector<double>          v(n), w(n, 7.0);

    // Scale A with 4 and multiply the scaled view with w
    v= 4 * A * w;
    std::cout << "v is " << v << "\n";

    // Scale w with 4 and multiply the scaled view with A
    v= A * (4 * w);
    std::cout << "v is " << v << "\n";

    // Scale both with 2 before multiplying
    v= 2 * A * (2 * w);
    std::cout << "v is " << v << "\n";

    // Scale A before multiplying and scale it back afterwards
    A*= 4;
    v= A * w;
    A*= 0.25;
    std::cout << "v is " << v << "\n";

    // Scale w before multiplying and scale it back afterwards
    w*= 4;
    v= A * w;
    w*= 0.25;
    std::cout << "v is " << v << "\n";

    return 0;
}
