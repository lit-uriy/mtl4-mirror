// $COPYRIGHT$

// Ugly workaround for ugly compiler bug
#define CONCEPTS_WITHOUT_OVERLOADED_REQUIREMENTS

#include <iostream>
#include <boost/test/minimal.hpp>
#include <concepts>

#include <boost/numeric/complex/complex.hpp>
#include <boost/numeric/mtl/mtl.hpp>



int test_main(int argc, char* argv[])
{
    using newstd::complex;
    const complex<float> iu(0., 1.), one(1.);

    float array1[][2]= {{1., 2.}, {3., 4.}}, array2[][2]= {{5., 6.}, {7., 8.}};
    mtl::dense2D<float> M1(array1), M2(array2);

    // Needs further work
    // mtl::dense2D<complex<float> >   A(M1 + iu * M2); // complex matrix
    std::cout << "A is\n" << A;



    return 0;
}
