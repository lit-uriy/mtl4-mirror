// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>

// #define MTL_HAS_BLAS
// #define MTL_USE_OPTERON_OPTIMIZATION

#include <boost/numeric/mtl/vector/dense_vector.hpp>


using namespace mtl;
using namespace std;  


template <typename VectorU, typename VectorV, typename VectorW>
void test(VectorU& u, VectorV& v, VectorW& w, const char* name)
{
    
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

}
 

int test_main(int argc, char* argv[])
{
    dense_vector<float>   u(5), v(5), w(5);
    dense_vector<double>  x(5), y(5), z(5);

    std::cout << "Testing vector operations\n";

    return 0;
}
 














