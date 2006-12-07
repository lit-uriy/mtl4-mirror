// $COPYRIGHT$

#ifndef MTL_SPECIALIZED_MATRIX_MULT_INCLUDE
#define MTL_SPECIALIZED_MATRIX_MULT_INCLUDE

#include <boost/numeric/mtl/operations/set_to_0.hpp>
#include <boost/numeric/mtl/operations/specialize_mult_type.hpp>
#include <boost/numeric/mtl/operations/matrix_mult.hpp>
#include <boost/numeric/mtl/recursion/recursive_matrix_mult.hpp>
#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>
#include <boost/numeric/mtl/recursion/base_case_matrix.hpp>


namespace mtl {

template <typename MatrixA, typename MatrixB, typename MatrixC>
void specialized_mult_add(MatrixA const& a, MatrixB const& b, MatrixC& c) 
{
    typedef recursion::bound_test_static<32>                      BaseCaseTest;

    using recursion::base_case_matrix;
    typedef typename base_case_matrix<MatrixA, BaseCaseTest>::type base_a_type;
    typedef typename base_case_matrix<MatrixB, BaseCaseTest>::type base_b_type;
    typedef typename base_case_matrix<MatrixC, BaseCaseTest>::type base_c_type;

    typedef typename mtl::specialize_mult_type<
        MatrixA, MatrixB, MatrixC
      , BaseCaseTest
      , functor::mult_add_simple_t<base_a_type, base_b_type, base_c_type>
    >::type                                                       mult_type;

    // std::cout << "Mult type " << typeid(mult_type()).name() << "\n";

    using recursion::matrix_recurator;
    matrix_recurator<MatrixA>    rec_a(a);
    matrix_recurator<MatrixB>    rec_b(b);
    matrix_recurator<MatrixC>    rec_c(c);
    equalize_depth(rec_a, rec_b, rec_c);

    using recursion::recurator_mult_add;
    recurator_mult_add(rec_a, rec_b, rec_c, mult_type(), BaseCaseTest());
}
 
template <typename MatrixA, typename MatrixB, typename MatrixC>
void specialized_matrix_mult(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    set_to_0(c);
    specialized_mult_add(a, b, c);
}


} // namespace mtl

#endif // MTL_SPECIALIZED_MATRIX_MULT_INCLUDE
