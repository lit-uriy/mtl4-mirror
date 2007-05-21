// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/vector/parameter.hpp>
#include <boost/numeric/mtl/operation/operators.hpp>

using namespace mtl;
using namespace std;  


template <typename Vector>
void one_d_iteration(char const* name, Vector & vector, size_t check_index, double check)
{
    typename traits::index<Vector>::type                               index(vector);
    typename traits::value<Vector>::type                               value(vector); 
    typedef  glas::tag::nz                                             tag;
    typedef typename traits::range_generator<tag, Vector>::type        cursor_type;
    typedef typename traits::range_generator<tag, Vector>::complexity  complexity;

    cout << name << "\nElements: " << complexity() << '\n';
    for (cursor_type cursor = begin<tag>(vector), cend = end<tag>(vector); cursor != cend; ++cursor) {
	cout << "vector[" << index(*cursor) << "] = " << value(*cursor) << '\n';
	if (index(*cursor) == check_index && value(*cursor) != check) 
	    throw "wrong check value";
    }
}
    

template <typename VectorU, typename VectorV, typename VectorW>
void test(VectorU& u, VectorV& v, VectorW& w, const char* name)
{
    // u= (typename VectorU::value_type)(3.0); 
    // v= (typename VectorV::value_type)(4.0); 
    // w= (typename VectorW::value_type)(5.0); 
    mtl::vector::fill(u, 3.0);
    mtl::vector::fill(v, 4.0);
    mtl::vector::fill(w, 5.0);

    std::cout << "\n\n";
    one_d_iteration(name, u, 2, (typename VectorU::value_type)(3.0));

    std::cout << name << "  --- u= v + w:\n"; std::cout.flush();
    u= v + w;
    cout << "u: " << u << "\n"; std::cout.flush();
    if (u[0] != 9.0) throw "wrong";

    std::cout << name << "  --- u= v + w + v + w:\n"; std::cout.flush();
    u= v + w + v + w;
    cout << "u: " << u << "\n"; std::cout.flush();
    if (u[0] != 18.0) throw "wrong";

    std::cout << name << "  --- u= w + (v= w + w);:\n"; std::cout.flush();
    u= w + (v= w + w);
    cout << "u: " << u << "v: " << v << "\n"; std::cout.flush();
    if (v[0] != 10.0) throw "v wrong";
    if (u[0] != 15.0) throw "u wrong";

    mtl::vector::fill(v, 4.0);
    mtl::vector::fill(w, 5.0);
    std::cout << name << "  --- u= (v= w + w) + v;:\n"; std::cout.flush();
    u= (v= w + w) + v;
    cout << "u: " << u << "v: " << v << "\n"; std::cout.flush();
    if (v[0] != 10.0) throw "v wrong";
    if (u[0] != 20.0) throw "u wrong";

    mtl::vector::fill(w, 4.0);
    std::cout << name << "  --- w= 4; u-= (v= w + w) - w;:\n"; std::cout.flush();
    u-= (v= w + w) - w;
    cout << "u: " << u << "v: " << v << "\n"; std::cout.flush();
    if (v[0] != 8.0) throw "v wrong";
    if (u[0] != 16.0) throw "u wrong"; // for -=

    
    std::cout << name << "  --- v= 3*u + 4*w;:\n"; std::cout.flush();
    v= 3*u + 4*w;
    cout << "v: " << v << "\n"; std::cout.flush();
    if (v[0] != 64.0) throw "v wrong";

    mtl::vector::fill(u, 3.0);
    mtl::vector::fill(v, 4.0);
    mtl::vector::fill(w, 5.0);
    std::cout << name << "  --- u= 3; v= 4; w=5; u+= v+= 3.0 * w;:\n"; std::cout.flush();
    u+= v+= 3.0 * w;
    cout << "u: " << u << "v: " << v << "\n"; std::cout.flush();
    if (v[0] != 19.0) throw "v wrong";
    if (u[0] != 22.0) throw "u wrong";

    mtl::vector::fill(u, 3.0);
    mtl::vector::fill(v, 4.0);
    mtl::vector::fill(w, 5.0);
    std::cout << name << "  --- u= 3; v= 4; w=5; u+= (v*= 3.0) + (w*= 2.0);:\n"; std::cout.flush();
    u+= (v*= 3.0) + (w*= 2.0);
    cout << "u: " << u << "v: " << v << "w: " << w << "\n"; std::cout.flush();
    if (v[0] != 12.0) throw "v wrong";
    if (w[0] != 10.0) throw "v wrong";
    if (u[0] != 25.0) throw "u wrong";
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
 














