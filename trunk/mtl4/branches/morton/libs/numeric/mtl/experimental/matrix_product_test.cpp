// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>
// #include <boost/numeric_cast.hpp>

#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/operations/set_to_0.hpp>
#include <boost/numeric/mtl/range_generator.hpp>

using namespace mtl;
using namespace std;  


/*
- Check matrix product C = A * B with:
  - A is MxN, B is NxL, C is MxL
  - with matrices a_ij = i+j, b_ij = 2(i+j); 
  - c_ij = 1/3 N (1 - 3i - 3j + 6ij - 3N + 3iN + 3jN + 2N^2).

*/

// Not really generic
template <typename Value>
double result_i_j (Value i, Value j, Value N)
{
    return 1.0/3.0 * N * (1.0 - 3*i - 3*j + 6*i*j - 3*N + 3*i*N + 3*j*N + 2*N*N);
}


// Only to be used for dense matrices
// Would work on sparse matrices with inserter but would be very expensive
template <typename Matrix, typename Value>
void fill_matrix(Matrix& matrix, Value factor)
{
    typedef typename Matrix::value_type    value_type;
    typedef typename Matrix::size_type     size_type;
    for (size_type r= matrix.begin_row(); r < matrix.end_row(); r++)
	for (size_type c= matrix.begin_col(); c < matrix.end_col(); c++)
	    matrix[r][c]= factor * (value_type(r) + value_type(c));
}


namespace functor {

    template <typename MatrixA, typename MatrixB, typename MatrixC>
    struct matrix_mult_variations
    {
        // set_to_0(c);
	using glas::tags::row_t; using glas::tags::col_t; using glas::tags::all_t;

        typedef typename traits::const_value<MatrixA>::type                a_value_type;
        typedef typename traits::const_value<MatrixB>::type                b_value_type;
        typedef typename traits::value<MatrixA>::type                      c_value_type;

        typedef typename traits::range_generator<row_t, MatrixA>::type     a_cur_type;
        typedef typename traits::range_generator<row_t, MatrixC>::type     c_cur_type;
        
        typedef typename traits::range_generator<col_t, MatrixB>::type     b_cur_type;
        typedef typename traits::range_generator<all_t, c_cur_type>::type  c_icur_type;

        typedef typename traits::range_generator<all_t, a_cur_type>::type  a_icur_type;
        typedef typename traits::range_generator<all_t, b_cur_type>::type  b_icur_type;

        void mult_add_simple(MatrixA const& a, MatrixB const& b, MatrixC& c)
        {
	    using glas::tags::row_t; using glas::tags::col_t; using glas::tags::all_t;
    		
            a_cur_type ac= begin<row_t>(a), aend= end<row_t>(a);
            for (c_cur_type cc= begin<row_t>(c); ac != aend; ++ac, ++cc) {

		b_cur_type bc= begin<col_t>(b), bend= end<col_t>(b);
		for (c_icur_type cic= begin<all_t>(cc); bc != bend; ++bc, ++cic) { 
		    
		    typename MatrixC::value_type c_tmp(c_value(*cic));
		    a_icur_type aic= begin<all_t>(ac), aiend= end<all_t>(ac); 
		    for (b_icur_type bic= begin<all_t>(bc); aic != aiend; ++aic, ++bic)
			c_tmp+= a_value(*aic) * b_value(*bic);
		    c_value(*cic, c_tmp);
		}
	    }
        }
    }

        
    template <typename MatrixA, typename MatrixB, typename MatrixC>
    struct mult_add_simple_t
    {
	void operator(MatrixA const& a, MatrixB const& b, MatrixC& c)
	{
	    matrix_mult_variations<MatrixA, MatrixB, MatrixC>().mult_add_simple(a, b, c);
	}

    };

} // namespace functor 


template <typename MatrixA, typename MatrixB, typename MatrixC>
void mult_add_simple(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    set_to_0(c);
    functor::mult_add_simple_t<MatrixA, MatrixB, MatrixC>()(a, b, c);
}


template <typename Value>
inline bool similar_values(Value x, Value y) 
{
    using std::abs; using std::max;
    return abs(x - y) / max(abs(x), abs(y)) < 0.000001;
}


// Check if matrix c is a * b according to convention above
// C has dimensions M x L and reduced_dim is N, see above
// A, B, and C are supposed to have the same indices: either all starting  from 0 or all from 1
template <typename Matrix>
void check_matrix_product(Matrix const& c, typename Matrix::size_type reduced_dim)
{
    typedef typename Matrix::value_type    value_type;
    typedef typename Matrix::size_type     size_type;
    size_type  rb= c.begin_row(), rl= c.end_row() - 1,
               cb= c.begin_col(), cl= c.end_col() - 1;

    if (!similar_values(value_type(result_i_j(rb, cb, reduced_dim)), c[rb][cb])) {
	cout << "Result in c[" << rb << "][" << cb << "] should be " << result_i_j(rb, cb, reduced_dim)
	     << " but is " << c[rb][cb] << "\n";
	throw "Wrong result"; }

    if (!similar_values(value_type(result_i_j(rl, cb, reduced_dim)), c[rl][cb])) {
	cout << "Result in c[" << rl << "][" << cb << "] should be " << result_i_j(rl, cb, reduced_dim)
	     << " but is " << c[rl][cb] << "\n";
	throw "Wrong result"; }

    if (!similar_values(value_type(result_i_j(rb, cl, reduced_dim)), c[rb][cl])) {
	cout << "Result in c[" << rb << "][" << cb << "] should be " << result_i_j(rb, cl, reduced_dim)
	     << " but is " << c[rb][cl] << "\n";
	throw "Wrong result"; }

    if (!similar_values(value_type(result_i_j(rl, cl, reduced_dim)), c[rl][cl])) {
	cout << "Result in c[" << rl << "][" << cb << "] should be " << result_i_j(rl, cl, reduced_dim)
	     << " but is " << c[rl][cl] << "\n";
	throw "Wrong result"; }

    // In the center of the matrix
    if (!similar_values(value_type(result_i_j((rb+rl)/2, (cb+cl)/2, reduced_dim)), c[(rb+rl)/2][(cb+cl)/2])) {
	cout << "Result in c[" << (rb+rl)/2 << "][" << (cb+cl)/2 << "] should be " << result_i_j((rb+rl)/2, (cb+cl)/2, reduced_dim)
	     << " but is " << c[(rb+rl)/2][(cb+cl)/2] << "\n";
	throw "Wrong result"; }
}








int test_main(int argc, char* argv[])
{
    //morton_dense<double,  0x55555555>      mda(3, 7), mdb(7, 2), mdc(3, 2);
    morton_dense<double,  0x55555555>      mda(5, 7), mdb(7, 6), mdc(5, 6);
    fill_matrix(mda, 1.0); fill_matrix(mdb, 2.0);
    cout << "mda:\n";    print_matrix_row_cursor(mda);
    cout << "\nmdb:\n";  print_matrix_row_cursor(mdb);

    matrix_mult_simple(mda, mdb, mdc);
    cout << "\nmdc:\n";  print_matrix_row_cursor(mdc);
    check_matrix_product(mdc, 7);

    mtl::dense2D<double> da(5, 7), db(7, 6), dc(5, 6);
    fill_matrix(da, 1.0); fill_matrix(db, 2.0);
    cout << "\nda:\n";   print_matrix_row_cursor(da);
    cout << "\ndb:\n";   print_matrix_row_cursor(db);

    matrix_mult_simple(da, db, dc);
    cout << "\ndc:\n";   print_matrix_row_cursor(dc);
    check_matrix_product(dc, 7);

    return 0;
}




#if 0

// Pure function implementation, for reference purposes kept for a while
template <typename MatrixA, typename MatrixB, typename MatrixC>
void matrix_mult_simple(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    using glas::tags::row_t; using glas::tags::col_t; using glas::tags::all_t;

    set_to_0(c);

    typename traits::const_value<MatrixA>::type                        a_value(a);
    typename traits::const_value<MatrixB>::type                        b_value(b);
    typename traits::value<MatrixA>::type                              c_value(c);

    typedef typename traits::range_generator<row_t, MatrixA>::type     a_cur_type;
    typedef typename traits::range_generator<row_t, MatrixC>::type     c_cur_type;
    
    typedef typename traits::range_generator<col_t, MatrixB>::type     b_cur_type;
    typedef typename traits::range_generator<all_t, c_cur_type>::type  c_icur_type;

    typedef typename traits::range_generator<all_t, a_cur_type>::type  a_icur_type;
    typedef typename traits::range_generator<all_t, b_cur_type>::type  b_icur_type;

    a_cur_type ac= begin<row_t>(a), aend= end<row_t>(a);
    for (c_cur_type cc= begin<row_t>(c); ac != aend; ++ac, ++cc) {

	b_cur_type bc= begin<col_t>(b), bend= end<col_t>(b);
	for (c_icur_type cic= begin<all_t>(cc); bc != bend; ++bc, ++cic) { 

	    typename MatrixC::value_type c_tmp(c_value(*cic));
	    a_icur_type aic= begin<all_t>(ac), aiend= end<all_t>(ac); 
	    for (b_icur_type bic= begin<all_t>(bc); aic != aiend; ++aic, ++bic)
		c_tmp+= a_value(*aic) * b_value(*bic);
	    c_value(*cic, c_tmp);
	}
    }
}
#endif
