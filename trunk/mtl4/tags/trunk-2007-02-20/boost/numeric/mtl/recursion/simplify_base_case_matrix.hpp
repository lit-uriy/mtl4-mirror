// $COPYRIGHT$

#ifndef MTL_SIMPLIFY_BASE_CASE_MATRIX_INCLUDE
#define MTL_SIMPLIFY_BASE_CASE_MATRIX_INCLUDE

#include <boost/numeric/mtl/dimensions.hpp>
#include <boost/numeric/mtl/operations/sub_matrix.hpp>
#include <boost/numeric/mtl/recursion/base_case_matrix.hpp>


namespace mtl { namespace recursion {


namespace impl {

    // With conversion, i.e. when target and source type are different
    template <typename Matrix, typename BaseCaseMatrix, typename BaseCaseTest>
    struct simplify_base_case_matrix
    {
	BaseCaseMatrix operator()(Matrix const& matrix, BaseCaseTest const test)
	{
	    typename Matrix::size_type begin_row= matrix.begin_row(), begin_col= matrix.begin_col();

	    if (matrix.num_rows() == BaseCaseTest::base_case_size 
		&& matrix.num_cols() == BaseCaseTest::base_case_size)
		return  BaseCaseMatrix(non_fixed::dimensions(matrix.num_rows(), matrix.num_cols()),
				       &const_cast<Matrix&>(matrix)[begin_row][begin_col]); 
	    
	    BaseCaseMatrix square(non_fixed::dimensions(BaseCaseTest::base_case_size, BaseCaseTest::base_case_size),
				  &const_cast<Matrix&>(matrix)[begin_row][begin_col]); 
	    return sub_matrix(square, begin_row, matrix.num_rows(), begin_col, matrix.num_cols());
	}
    };
      
    template <typename Matrix, typename BaseCaseTest>
    struct simplify_base_case_matrix<Matrix, Matrix, BaseCaseTest>
    {
	Matrix operator()(Matrix const& matrix, BaseCaseTest const test)
	{
	    return matrix;
	}
    };


#if 0
    inline BaseCaseMatrix 
    simplify_base_case_matrix(Matrix const& matrix, BaseCaseMatrix const&, BaseCaseTest const&)
    {
	typename Matrix::size_type begin_row= matrix.begin_row(), begin_col= matrix.begin_col();

	if (matrix.num_rows() == BaseCaseTest::base_case_size 
	    && matrix.num_cols() == BaseCaseTest::base_case_size)
	    return  BaseCaseMatrix(non_fixed::dimensions(matrix.num_rows(), matrix.num_cols()),
				   &const_cast<Matrix&>(matrix)[begin_row][begin_col]); 
		
	BaseCaseMatrix square(non_fixed::dimensions(BaseCaseTest::base_case_size, BaseCaseTest::base_case_size),
			      &const_cast<Matrix&>(matrix)[begin_row][begin_col]); 
	return sub_matrix(square, begin_row, matrix.num_rows(), begin_col, matrix.num_cols());
    }

    // Without conversion, i.e. when target and source type are identical
    template <typename Matrix, typename BaseCaseTest>
    inline Matrix 
    simplify_base_case_matrix(Matrix const& matrix, Matrix const&, BaseCaseTest const&)
    {
	return matrix;
    }
#endif

} // namespace impl

template <typename Matrix, typename BaseCaseTest>
typename base_case_matrix<Matrix, BaseCaseTest>::type inline
simplify_base_case_matrix(Matrix const& matrix, BaseCaseTest test)
{
    // cout << "simplify dim " <<  matrix.num_rows() << ", " << matrix.num_cols() << "\n";
    if (matrix.num_rows() > BaseCaseTest::base_case_size 
	|| matrix.num_cols() > BaseCaseTest::base_case_size)  {
      throw "Matrix dimension is larger than base case";
    }
    return impl::simplify_base_case_matrix<Matrix, typename base_case_matrix<Matrix, BaseCaseTest>::type, BaseCaseTest>()(matrix, test);

}

}} // namespace mtl::recursion


#endif // MTL_SIMPLIFY_BASE_CASE_MATRIX_INCLUDE
