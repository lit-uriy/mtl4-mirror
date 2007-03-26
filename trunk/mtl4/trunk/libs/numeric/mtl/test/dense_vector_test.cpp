// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>


using namespace mtl;
using namespace std;  


template <typename VectorU, typename VectorV, typename VectorW>
void test(VectorU& u, VectorV& v, VectorW& w, const char* name)
{
#if 0
    u= 3.0; v= 4.0; w= 5.0;

    std::cout << "\n" << name << "  --- u= v + w:\n"; std::cout.flush();
    u= v + w;
    if (u[0] != 9.0) throw "wrong";

    std::cout << "\n" << name << "  --- u= v + w + v + w:\n"; std::cout.flush();
    u= v + w + v + w;
    if (u[0] != 18.0) throw "wrong";

    std::cout << "\n" << name << "  --- u= v + v= w + w;:\n"; std::cout.flush();
    u= (v + (v= w + w));
    if (v[0] != 10.0) throw "v wrong";
    if (w[0] != 20.0) throw "w wrong";
#endif
}
 

int test_main(int argc, char* argv[])
{
    dense_vector<float>   u(5), v(5), w(5);
    dense_vector<double>  x(5), y(5), z(5);

    std::cout << "Testing vector operations\n";

    u[0]= 1.0, u[1]= 2.0, u[2]= 3.0, u[3]= 5.0, u[4]= 6.0;
    cout << "u element-wise assigned" << u << "\n";

    u= 3.0;
    cout << "u assigned from scalar" << u << "\n";


    test(u, v, v, "test float");
    test(x, y, z, "test double");
    test(u, x, y, "test float, double mixed");

    return 0;
}
 














