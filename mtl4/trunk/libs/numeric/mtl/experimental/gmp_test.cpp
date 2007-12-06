#include <gmpxx.h>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

// Check only if malloc error happens as reported by Hui Li

int test_main(int argc, char** argv) 
{
    mtl::dense2D<mpz_class> A;
    return 0;
}



// g++ -g -DMTL_ASSERT_FOR_THROW -I$MTL_BOOST_ROOT -I/home/pgottsch/Download/gmp-4.2.2 -I/home/pgottsch/projects/boost/boost_1_33_1 -o gmp_test gmp_test.cpp -L/home/pgottsch/Download/gmp-4.2.2 -lgmp

