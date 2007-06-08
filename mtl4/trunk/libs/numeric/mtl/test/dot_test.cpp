// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>


#include <boost/numeric/mtl/mtl.hpp>

using namespace mtl;
using namespace std;  


template <typename VectorU, typename VectorV>
void test(VectorU& u, VectorV& v, const char* name)
{
    for (int i= 0; i < size(v); i++)
	u[i]= i+1, v[i]= i+1;

    std::cout << name << "\n dot(u, v) = " << dot(u, v) << "\n"; std::cout.flush();
    if (dot(u, v) != 285.0) throw "dot product wrong";

    std::cout << " dot<2>(u, v) = " << dot<2>(u, v) << "\n"; std::cout.flush();
    if (dot<2>(u, v) != 285.0) throw "dot product wrong";

    std::cout << " dot<6>(u, v) = " << dot<6>(u, v) << "\n"; std::cout.flush();
    if (dot<6>(u, v) != 285.0) throw "dot product wrong";
}
 

int test_main(int argc, char* argv[])
{
    using mtl::vector::parameters;
    const int size= 9;

    dense_vector<float>   u(size), v(size), w(size);
    dense_vector<double>  x(size), y(size), z(size);
    dense_vector<std::complex<double> >  xc(size), yc(size), zc(size);

    std::cout << "Testing vector operations\n";

    test(u, v, "test float");
    test(x, y, "test double");
    test(u, x, "test float, double mixed");
    test(xc, yc, "test complex<double>");
    test(x, yc, "test complex<double>, double mixed");

    dense_vector<float, parameters<row_major> >   ur(size), vr(size), wr(size);
    test(ur, vr, "test float in row vector");
    
    // test(ur, v, wr, "test float in mixed vector (shouldn't work)"); 

    return 0;
}
 














