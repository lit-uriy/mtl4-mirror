// $COPYRIGHT$

#ifndef MTL_HESSIAN_MATRIX_UTILITIES_INCLUDE
#define MTL_HESSIAN_MATRIX_UTILITIES_INCLUDE

#include <cmath>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>

namespace mtl {

    namespace matrix {
	// Fills a matrix A with a_ij = factor * (i + j)
	// Only to be used for dense matrices
	// Would work on sparse matrices with inserter but would be very expensive
	template <typename Matrix, typename Value>
	void hessian_setup(Matrix& matrix, Value factor)
	{
	    typedef typename Matrix::value_type    value_type;
	    typedef typename Matrix::size_type     size_type;
	    for (size_type r= matrix.begin_row(); r < matrix.end_row(); r++)
		for (size_type c= matrix.begin_col(); c < matrix.end_col(); c++)
		    matrix[r][c]= factor * (value_type(r) + value_type(c));
	}
    }

    using matrix::hessian_setup;

namespace impl {

    /*
    - Check matrix product C = A * B with:
      - A is MxN, B is NxL, C is MxL
      - with matrices a_ij = i+j, b_ij = 2(i+j); 
      - c_ij = 1/3 N (1 - 3i - 3j + 6ij - 3N + 3iN + 3jN + 2N^2).

    */
    // Not really generic
    template <typename Value>
    double inline hessian_product_i_j (Value i, Value j, Value N)
    {
        return 1.0/3.0 * N * (1.0 - 3*i - 3*j + 6*i*j - 3*N + 3*i*N + 3*j*N + 2*N*N);
    }
        
    template <typename Value>
    inline bool similar_values(Value x, Value y) 
    {
        using std::abs; using std::max;
        return abs(x - y) / max(abs(x), abs(y)) < 0.000001;
    }
} // impl       


// Check if matrix c is a * b according to convention above
// C has dimensions M x L and reduced_dim is N, see above
// A, B, and C are supposed to have the same indices: either all starting  from 0 or all from 1
template <typename Matrix>
void check_hessian_matrix_product(Matrix const& c, typename Matrix::size_type reduced_dim, double factor= 1.0)
{
    using impl::similar_values; using impl::hessian_product_i_j;
    typedef typename Matrix::value_type    value_type;
    typedef typename Matrix::size_type     size_type;
    size_type  rb= c.begin_row(), rl= c.end_row() - 1,
               cb= c.begin_col(), cl= c.end_col() - 1;

    if (!similar_values(value_type(factor * hessian_product_i_j(rb, cb, reduced_dim)), c[rb][cb])) {
	std::cout << "Result in c[" << rb << "][" << cb << "] should be " 
		  << factor * hessian_product_i_j(rb, cb, reduced_dim)
		  << " but is " << c[rb][cb] << "\n";
	MTL_THROW(not_expected_result());
    }

    if (!similar_values(value_type(factor * hessian_product_i_j(rl, cb, reduced_dim)), c[rl][cb])) {
	std::cout << "Result ixn c[" << rl << "][" << cb << "] should be " 
		  << factor * hessian_product_i_j(rl, cb, reduced_dim)
		  << " but is " << c[rl][cb] << "\n";
	MTL_THROW(not_expected_result()); 
    }

    if (!similar_values(value_type(factor * hessian_product_i_j(rb, cl, reduced_dim)), c[rb][cl])) {
	std::cout << "Result in c[" << rb << "][" << cb << "] should be " 
		  << factor * hessian_product_i_j(rb, cl, reduced_dim)
		  << " but is " << c[rb][cl] << "\n";
	MTL_THROW(not_expected_result());
    }

    if (!similar_values(value_type(factor * hessian_product_i_j(rl, cl, reduced_dim)), c[rl][cl])) {
	std::cout << "Result in c[" << rl << "][" << cb << "] should be " 
		  << factor * hessian_product_i_j(rl, cl, reduced_dim)
		  << " but is " << c[rl][cl] << "\n";
	MTL_THROW(not_expected_result()); 
    }

    // In the center of the matrix
    if (!similar_values(value_type(factor * hessian_product_i_j((rb+rl)/2, (cb+cl)/2, reduced_dim)), 
			c[(rb+rl)/2][(cb+cl)/2])) {
	std::cout << "Result in c[" << (rb+rl)/2 << "][" << (cb+cl)/2 << "] should be " 
		  << factor * hessian_product_i_j((rb+rl)/2, (cb+cl)/2, reduced_dim)
		  << " but is " << c[(rb+rl)/2][(cb+cl)/2] << "\n";
	MTL_THROW(not_expected_result()); 
    }
}


} // namespace mtl

#endif // MTL_HESSIAN_MATRIX_UTILITIES_INCLUDE
