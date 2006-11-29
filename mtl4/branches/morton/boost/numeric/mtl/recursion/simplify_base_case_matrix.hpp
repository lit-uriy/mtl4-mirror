// $COPYRIGHT$

#ifndef MTL_SIMPLIFY_BASE_CASE_MATRIX_INCLUDE
#define MTL_SIMPLIFY_BASE_CASE_MATRIX_INCLUDE

#include <boost/numeric/mtl/dimensions.hpp>


namespace mtl { namespace recursion {


namespace impl {

    // Without conversion, i.e. when target and source type are identical
    template <typename Matrix>
    inline Matrix 
    simplify_base_case_matrix(Matrix const& matrix, Matrix const&)
    {
	return matrix;
    }

    // With conversion, i.e. when target and source type are different
    template <typename Matrix, typename BaseCaseMatrix>
    inline BaseCaseMatrix 
    simplify_base_case_matrix(Matrix const& matrix, BaseCaseMatrix const&)
    {
	return BaseCaseMatrix(non_fixed::dimensions(matrix.num_rows(), matrix.num_cols()),
			      &const_cast<Matrix&>(matrix)[matrix.begin_row()][matrix.begin_col()]);
    }


} // namespace impl

template <typename Matrix, typename BaseCaseTest>
typename base_case_matrix<Matrix, BaseCaseTest>::type inline
simplify_base_case_matrix(Matrix const& matrix, BaseCaseTest const&)
{
    // cout << "simplify dim " <<  matrix.num_rows() << ", " << matrix.num_cols() << "\n";
    if (matrix.num_rows() != BaseCaseTest::base_case_size 
	|| matrix.num_cols() != BaseCaseTest::base_case_size) throw "Base case and matrix have different dimensions";
    return impl::simplify_base_case_matrix(matrix, typename base_case_matrix<Matrix, BaseCaseTest>::type());
}

}} // namespace mtl::recursion


#endif // MTL_SIMPLIFY_BASE_CASE_MATRIX_INCLUDE
