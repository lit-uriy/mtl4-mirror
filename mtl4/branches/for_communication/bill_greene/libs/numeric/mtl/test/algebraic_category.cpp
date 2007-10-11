// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>

#include <boost/numeric/mtl/recursion/bit_masking.hpp>
#include <boost/numeric/mtl/recursion/predefined_masks.hpp>
#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/matrix/morton_dense.hpp> 
#include <boost/numeric/mtl/matrix/compressed2D.hpp> 
#include <boost/numeric/mtl/matrix/map_view.hpp>
#include <boost/numeric/mtl/matrix/parameter.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/vector/parameter.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>


using namespace mtl;
using namespace std;  

template <typename T>
void dispatch(const T& x, tag::scalar, const char* name)
{
    cout << "dispatch " << name << " to scalar\n";
}

template <typename T>
void dispatch(const T& x, tag::vector, const char* name)
{
    cout << "dispatch " << name << " to vector\n";
}

template <typename T>
void dispatch(const T& x, tag::matrix, const char* name)
{ 
    cout << "dispatch " << name << " to matrix\n";
}


template <typename T>
void test(const T& x, const char* name)
{
    dispatch(x, typename traits::algebraic_category<T>::type(), name);
}

struct none {};

int test_main(int argc, char* argv[])
{
    const int size= 5;
    double d;
    int    i;
    none   n;

    dense_vector<float>                                  u(size);
    dense_vector<float, mtl::vector::parameters<row_major> >  ur(size);

    dense2D<double>                                      dr(size, size);
    dense2D<double, matrix::parameters<col_major> >      dc(size, size);
    morton_dense<double, recursion::morton_z_mask>       mzd(size, size);
    morton_dense<double, recursion::doppled_2_row_mask>  d2r(size, size);
    compressed2D<double>                                 cr(size, size);
    compressed2D<double, matrix::parameters<col_major> > cc(size, size);

    dense2D<complex<double> >                            drc(size, size);
    compressed2D<complex<double> >                       crc(size, size);

    matrix::scaled_view<double, dense2D<double> >        scaled_matrix(2.0, dr);

    test(d, "double");
    test(i, "int");
    test(n, "unknown type");

    test(u, "dense (column) vector");
    test(ur, "dense row vector");

    test(dr, "Dense row major");
    test(dc, "Dense column major");
    test(mzd, "Morton Z-order");
    test(d2r, "Hybrid 2 row-major");
    test(cr, "Compressed row major");
    test(cc, "Compressed column major");
    test(drc, "Dense row major complex");
    test(crc, "Compressed row major complex");
    test(scaled_matrix, "Scaled_view of dense row major");


    return 0;
}
