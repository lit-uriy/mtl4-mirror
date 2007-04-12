// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/vector/parameter.hpp>

using namespace mtl;
using namespace std;  


template <typename VectorU, typename VectorV, typename VectorW>
void test(VectorU& u, VectorV& v, VectorW& w, const char* name)
{
    u= (typename VectorU::value_type)(3.0); 
    v= (typename VectorV::value_type)(4.0); 
    w= (typename VectorW::value_type)(5.0); 

    std::cout << "\n" << name << "  --- u= v + w:\n"; std::cout.flush();
    u= v + w;
    cout << "u: " << u << "\n"; std::cout.flush();
    if (u[0] != 9.0) throw "wrong";

    std::cout << "\n" << name << "  --- u= v + w + v + w:\n"; std::cout.flush();
    u= v + w + v + w;
    cout << "u: " << u << "\n"; std::cout.flush();
    if (u[0] != 18.0) throw "wrong";

    std::cout << "\n" << name << "  --- u= w + (v= w + w);:\n"; std::cout.flush();
    u= w + (v= w + w);
    cout << "u: " << u << "\n" << "v: " << v << "\n"; std::cout.flush();
    if (v[0] != 10.0) throw "v wrong";
    if (u[0] != 15.0) throw "u wrong";

    v= (typename VectorV::value_type)(4.0); 
    w= (typename VectorW::value_type)(5.0); 
    std::cout << "\n" << name << "  --- u= (v= w + w) + v;:\n"; std::cout.flush();
    u= (v= w + w) + v;
    cout << "u: " << u << "\n" << "v: " << v << "\n"; std::cout.flush();
    if (v[0] != 10.0) throw "v wrong";
    if (u[0] != 20.0) throw "u wrong";

    w= (typename VectorW::value_type)(4.0); 
    std::cout << "\n" << name << "  --- w= 4; u-= (v= w + w) - w;:\n"; std::cout.flush();
    u-= (v= w + w) - w;
    cout << "u: " << u << "\n" << "v: " << v << "\n"; std::cout.flush();
    if (v[0] != 8.0) throw "v wrong";
    if (u[0] != 16.0) throw "u wrong"; // for -=
}
 

int test_main(int argc, char* argv[])
{
    using mtl::vector::parameters;

    dense_vector<float>   u(5), v(5), w(5);
    dense_vector<double>  x(5), y(5), z(5);

    std::cout << "Testing vector operations\n";

    u[0]= 1.0, u[1]= 2.0, u[2]= 3.0, u[3]= 5.0, u[4]= 6.0;
    cout << "u element-wise assigned\n" << u << "\n";

    u= 3.0f;
    cout << "u assigned from scalar\n" << u << "\n";


    test(u, v, w, "test float");
    test(x, y, z, "test double");
    test(u, x, y, "test float, double mixed");

    dense_vector<float, parameters<row_major> >   ur(5), vr(5), wr(5);
    test(ur, vr, wr, "test float in row vector");
    
    // test(ur, v, wr, "test float in mixed vector (shouldn't work)"); 

    return 0;
}
 














