// $COPYRIGHT$

#include <iostream>
#include <cmath>
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
double inline result_i_j (Value i, Value j, Value N)
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

    template <unsigned MaxDepth, typename Value, typename Cursor1, typename Prop1, typename Cursor2, typename Prop2, unsigned Depth>
    struct cursor_pseudo_dot_block
    {
	static unsigned const offset= MaxDepth - Depth;
	
	void operator() (Cursor1 i1, Prop1& prop1, Cursor2 i2, Prop2& prop2,
			 Value& s0, Value& s1, Value& s2, Value& s3,
			 Value& s4, Value& s5, Value& s6, Value& s7)
	{
	    s0+= prop1(i1 + offset) * prop2(i2 + offset);
	    typedef cursor_pseudo_dot_block<MaxDepth, Value, Cursor1, Prop1, Cursor2, Prop2, Depth-1> block_rest;
	    block_rest() (i1, prop1, i2, prop2, s1, s2, s3, s4, s5, s6, s7, s0);
	}
    };

    //template <>
    template <unsigned MaxDepth, typename Value, typename Cursor1, typename Prop1, typename Cursor2, typename Prop2>
    struct cursor_pseudo_dot_block<MaxDepth, Value, Cursor1, Prop1, Cursor2, Prop2, 1>
    {
	static unsigned const offset= MaxDepth - 1;
	
	void operator() (Cursor1 i1, Prop1& prop1, Cursor2 i2, Prop2& prop2,
			 Value& s0, Value&, Value&, Value&,
			 Value&, Value&, Value&, Value&)
	{
	    s0+= prop1(i1 + offset) * prop2(i2 + offset);
	}
    };      

    template <unsigned MaxDepth, typename Value, typename Cursor1, typename Prop1, typename Cursor2, typename Prop2>
    struct cursor_pseudo_dot_t
    {
	Value operator() (Cursor1 i1, Cursor1 end1, Prop1& prop1, Cursor2 i2, Prop2& prop2)
	{
	    using math::zero;
	    Value         ref, my_zero(zero(ref)),
                          s0= my_zero, s1= my_zero, s2= my_zero, s3= my_zero, 
		          s4= my_zero, s5= my_zero, s6= my_zero, s7= my_zero;
	    std::size_t size= end1 - i1, blocks= size / MaxDepth, blocked_size= blocks * MaxDepth;

	    typedef cursor_pseudo_dot_block<MaxDepth, Value, Cursor1, Prop1, Cursor2, Prop2, MaxDepth> dot_block_type;
	    for (unsigned i= 0; i < blocked_size; i+= MaxDepth, i1+= MaxDepth, i2+= MaxDepth) {
		dot_block_type()(i1, prop1, i2, prop2, s0, s1, s2, s3, s4, s5, s6, s7);
	    }

	    typedef cursor_pseudo_dot_block<MaxDepth, Value, Cursor1, Prop1, Cursor2, Prop2, MaxDepth> dot_single_type;
	    s0+= s1 + s2 + s3 + s4 + s5 + s6 + s7;
	    for (unsigned i= blocked_size; i < size; ++i, ++i1, ++i2)
		dot_single_type()(i1, prop1, i2, prop2, s0, s1, s2, s3, s4, s5, s6, s7);
	    return s0;
	}
    };

} // namespace functor 

template <unsigned MaxDepth, typename Value, typename Cursor1, typename Prop1, typename Cursor2, typename Prop2>
Value cursor_pseudo_dot(Cursor1 i1, Cursor1 end1, Prop1 prop1, Cursor2 i2, Prop2 prop2, Value)
{
    return functor::cursor_pseudo_dot_t<MaxDepth, Value, Cursor1, Prop1, Cursor2, Prop2>()(i1, end1, prop1, i2, prop2);
}



namespace functor {

    template <typename MatrixA, typename MatrixB, typename MatrixC, unsigned DotUnroll= 8>
    struct matrix_mult_variations
    {
	// using glas::tags::row_t; using glas::tags::col_t; using glas::tags::all_t;
	typedef glas::tags::row_t                                          row_t;
	typedef glas::tags::col_t                                          col_t;
	typedef glas::tags::all_t                                          all_t;

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
	    a_value_type   a_value(a);
	    b_value_type   b_value(b);
	    c_value_type   c_value(c);
    		
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

	// template<unsigned DotUnroll>
        void mult_add_fast_dot(MatrixA const& a, MatrixB const& b, MatrixC& c)
        {
	    a_value_type   a_value(a);
	    b_value_type   b_value(b);
	    c_value_type   c_value(c);
    		
            a_cur_type ac= begin<row_t>(a), aend= end<row_t>(a);
            for (c_cur_type cc= begin<row_t>(c); ac != aend; ++ac, ++cc) {

		b_cur_type bc= begin<col_t>(b), bend= end<col_t>(b);
		for (c_icur_type cic= begin<all_t>(cc); bc != bend; ++bc, ++cic) { 
		    
		    a_icur_type aic= begin<all_t>(ac), aiend= end<all_t>(ac); 
		    b_icur_type bic= begin<all_t>(bc);
		    typename MatrixC::value_type c_tmp= c_value(*cic),
			dot_tmp= cursor_pseudo_dot<DotUnroll>(aic, aiend, a_value, bic, b_value, c_tmp);
		    c_value(*cic, c_tmp + dot_tmp);
		}		    
	    }
        }
    };

        
    template <typename MatrixA, typename MatrixB, typename MatrixC>
    struct mult_add_simple_t
    {
	void operator() (MatrixA const& a, MatrixB const& b, MatrixC& c)
	{
	    matrix_mult_variations<MatrixA, MatrixB, MatrixC>().mult_add_simple(a, b, c);
	}
    };

    template <typename MatrixA, typename MatrixB, typename MatrixC, unsigned DotUnroll= 8>
    struct mult_add_fast_dot_t
    {
	void operator() (MatrixA const& a, MatrixB const& b, MatrixC& c)
	{
	    matrix_mult_variations<MatrixA, MatrixB, MatrixC, DotUnroll> object;
	    object.mult_add_fast_dot(a, b, c);
	}
    };

} // namespace functor 


template <typename MatrixA, typename MatrixB, typename MatrixC>
void mult_add_simple(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    functor::mult_add_simple_t<MatrixA, MatrixB, MatrixC>()(a, b, c);
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void matrix_mult_simple(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    set_to_0(c);
    functor::mult_add_simple_t<MatrixA, MatrixB, MatrixC>()(a, b, c);
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void mult_add_fast_dot(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    functor::mult_add_fast_dot_t<MatrixA, MatrixB, MatrixC, 8>()(a, b, c);
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void matrix_mult_fast_dot(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    set_to_0(c);
    functor::mult_add_fast_dot_t<MatrixA, MatrixB, MatrixC, 8>()(a, b, c);
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

    cout << "\nNow with fast pseudo dot product\n\n";

#if 0
    matrix_mult_fast_dot(mda, mdb, mdc);
    cout << "\nmdc:\n";  print_matrix_row_cursor(mdc);
    check_matrix_product(mdc, 7);
#endif

    matrix_mult_fast_dot(da, db, dc);
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
