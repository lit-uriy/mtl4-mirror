// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>
// #include <boost/numeric_cast.hpp>

#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/operations/matrix_mult.hpp>
#include <boost/numeric/mtl/operations/hessian_matrix_utilities.hpp>

using namespace mtl;
using namespace std;  


int test_main(int argc, char* argv[])
{
    //morton_dense<double,  0x55555555>      mda(3, 7), mdb(7, 2), mdc(3, 2);
    morton_dense<double,  0x55555555>      mda(5, 7), mdb(7, 6), mdc(5, 6);
    fill_hessian_matrix(mda, 1.0); fill_hessian_matrix(mdb, 2.0);
    std::cout << "mda:\n";    print_matrix_row_cursor(mda);
    std::cout << "\nmdb:\n";  print_matrix_row_cursor(mdb);
 
    matrix_mult_simple(mda, mdb, mdc);
    std::cout << "\nmdc:\n";  print_matrix_row_cursor(mdc);
    check_hessian_matrix_product(mdc, 7);

    mtl::dense2D<double> da(5, 7), db(7, 6), dc(5, 6);
    fill_hessian_matrix(da, 1.0); fill_hessian_matrix(db, 2.0);
    std::cout << "\nda:\n";   print_matrix_row_cursor(da);
    std::cout << "\ndb:\n";   print_matrix_row_cursor(db);

    matrix_mult_simple(da, db, dc);
    std::cout << "\ndc:\n";   print_matrix_row_cursor(dc);
    check_hessian_matrix_product(dc, 7);

    std::cout << "\nNow with fast pseudo dot product\n\n";

#if 0
    matrix_mult_fast_dot(mda, mdb, mdc);
    std::cout << "\nmdc:\n";  print_matrix_row_cursor(mdc);
    check_hessian_matrix_product(mdc, 7);
#endif

    matrix_mult_fast_inner(da, db, dc);
    std::cout << "\ndc:\n";   print_matrix_row_cursor(dc);
    check_hessian_matrix_product(dc, 7);

    
    mtl::dense2D<double> da8(8, 8), db8(8, 8), dc8(8, 8);
    fill_hessian_matrix(da8, 1.0); fill_hessian_matrix(db8, 2.0);
    std::cout << "\nda8:\n";   print_matrix_row_cursor(da8);
    std::cout << "\ndb8:\n";   print_matrix_row_cursor(db8);

    matrix_mult_fast_middle(da8, db8, dc8);
    std::cout << "\ndc8:\n";   print_matrix_row_cursor(dc8);
    check_hessian_matrix_product(dc8, 8);

    matrix_mult_fast_middle(da, db, dc);
    std::cout << "\ndc:\n";   print_matrix_row_cursor(dc);
    check_hessian_matrix_product(dc, 7);

    matrix_mult_fast_outer(da8, db8, dc8);
    std::cout << "\ndc8:\n";   print_matrix_row_cursor(dc8);
    check_hessian_matrix_product(dc8, 8);

    matrix_mult_fast_outer(da, db, dc);
    std::cout << "\ndc:\n";   print_matrix_row_cursor(dc);
    check_hessian_matrix_product(dc, 7);

    matrix_mult(da8, db8, dc8);
    std::cout << "\ndc8:\n";   print_matrix_row_cursor(dc8);
    check_hessian_matrix_product(dc8, 8);

    matrix_mult<4>(da8, db8, dc8);
    std::cout << "\ndc8:\n";   print_matrix_row_cursor(dc8);
    check_hessian_matrix_product(dc8, 8);

    matrix_mult<4, 4>(da8, db8, dc8);
    std::cout << "\ndc8:\n";   print_matrix_row_cursor(dc8);
    check_hessian_matrix_product(dc8, 8);

    matrix_mult<4, 4, 4>(da8, db8, dc8);
    std::cout << "\ndc8:\n";   print_matrix_row_cursor(dc8);
    check_hessian_matrix_product(dc8, 8);

    return 0;
}




