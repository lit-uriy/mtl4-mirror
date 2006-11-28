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
double inline hessian_product_i_j (Value i, Value j, Value N)
{
    return 1.0/3.0 * N * (1.0 - 3*i - 3*j + 6*i*j - 3*N + 3*i*N + 3*j*N + 2*N*N);
}


// Only to be used for dense matrices
// Would work on sparse matrices with inserter but would be very expensive
template <typename Matrix, typename Value>
void fill_hessian_matrix(Matrix& matrix, Value factor)
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
	    Cursor1 tmp1(i1); tmp1+= offset;
	    Cursor2 tmp2(i2); tmp2+= offset;
	    s0+= prop1(*tmp1) * prop2(*tmp2);
	    // s0+= prop1(i1 + offset) * prop2(i2 + offset);
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
	    s0+= prop1(*(i1 + offset)) * prop2(*(i2 + offset));
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


template <typename MultiAction, unsigned MaxSteps, unsigned RemainingSteps>
struct multi_action_helper
{
    static unsigned const step= MaxSteps - RemainingSteps;

    void operator() (MultiAction const& action) const
    {
	action(step);
	multi_action_helper<MultiAction, MaxSteps, RemainingSteps-1>()(action);
    }

    void operator() (MultiAction& action) const
    {
	action(step);
	multi_action_helper<MultiAction, MaxSteps, RemainingSteps-1>()(action);
    }    
};


template <typename MultiAction, unsigned MaxSteps>
struct multi_action_helper<MultiAction, MaxSteps, 1>
{
    static unsigned const step= MaxSteps - 1;

    void operator() (MultiAction const& action) const
    {
	action(step);
    }

    void operator() (MultiAction& action) const
    {
	action(step);
    }    
};


template <typename MultiAction, unsigned Steps>
struct multi_action_block
{
    void operator() (MultiAction const& action) const
    {
	multi_action_helper<MultiAction, Steps, Steps>()(action);
    }

    void operator() (MultiAction& action) const
    {
	multi_action_helper<MultiAction, Steps, Steps>()(action);
    }
};

namespace functor {

    template <typename MatrixA, typename MatrixB, typename MatrixC, 
	      unsigned DotUnroll= 8, unsigned MiddleUnroll= 4, unsigned OuterUnroll= 1>
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

		a_icur_type aic= begin<all_t>(ac), aiend= end<all_t>(ac); // constant in inner loop
		b_cur_type bc= begin<col_t>(b), bend= end<col_t>(b);
		for (c_icur_type cic= begin<all_t>(cc); bc != bend; ++bc, ++cic) { 
		    
		    b_icur_type bic= begin<all_t>(bc);
		    typename MatrixC::value_type c_tmp= c_value(*cic),
			dot_tmp= cursor_pseudo_dot<DotUnroll>(aic, aiend, a_value, bic, b_value, c_tmp);
		    c_value(*cic, c_tmp + dot_tmp);
		}		    
	    }
        }


        void mult_add_fast_middle(MatrixA const& a, MatrixB const& b, MatrixC& c)
        {
	    if (b.num_rows() % MiddleUnroll != 0)
		throw "B's number of rows must be divisible by MiddleUnroll";

	    a_value_type   a_value(a);
	    b_value_type   b_value(b);
	    c_value_type   c_value(c);
    		
            a_cur_type ac= begin<row_t>(a), aend= end<row_t>(a);
            for (c_cur_type cc= begin<row_t>(c); ac != aend; ++ac, ++cc) {

		b_cur_type bc= begin<col_t>(b), bend= end<col_t>(b);
		for (c_icur_type cic= begin<all_t>(cc); bc != bend; bc+= MiddleUnroll, cic+= MiddleUnroll) { 

		    inner_block my_inner_block(a_value, b_value, c_value, ac, bc, cic);
		    multi_action_block<inner_block, MiddleUnroll>() (my_inner_block);
		}
	    }
        }

	struct inner_block
	{
	    inner_block(a_value_type const& a_value, b_value_type const& b_value, c_value_type& c_value, 
			a_cur_type const& ac, b_cur_type const& bc, c_icur_type const& cic) 
		: a_value(a_value), b_value(b_value), c_value(c_value), ac(ac), bc(bc), cic(cic)
	    {}

	    void operator()(unsigned step)
	    {
		// std::cout << "In inner_block: step " << step << "\n";
		b_cur_type my_bc= bc + step;
		c_icur_type my_cic= cic + step;

		typename MatrixC::value_type c_tmp(c_value(*my_cic));
		a_icur_type aic= begin<all_t>(ac), aiend= end<all_t>(ac); 
		for (b_icur_type bic= begin<all_t>(my_bc); aic != aiend; ++aic, ++bic)
		    c_tmp+= a_value(*aic) * b_value(*bic);
		c_value(*my_cic, c_tmp);
	    }
	private:
	    a_value_type const&   a_value;
	    b_value_type const&   b_value;
	    c_value_type&         c_value;
	    a_cur_type const&     ac;
	    b_cur_type const&     bc;
	    c_icur_type const&    cic;
	};
	

 	struct fast_inner_block
	{
	    fast_inner_block(a_value_type const& a_value, b_value_type const& b_value, c_value_type& c_value, 
			a_cur_type const& ac, b_cur_type const& bc, c_icur_type const& cic) 
		: a_value(a_value), b_value(b_value), c_value(c_value), ac(ac), bc(bc), cic(cic)
	    {}

	    void operator()(unsigned step)
	    {
		// std::cout << "In inner_block: step " << step << "\n";
		b_cur_type my_bc= bc + step;
		c_icur_type my_cic= cic + step;

		a_icur_type aic= begin<all_t>(ac), aiend= end<all_t>(ac); 
		b_icur_type bic= begin<all_t>(my_bc);
		typename MatrixC::value_type c_tmp= c_value(*my_cic),
		    dot_tmp= cursor_pseudo_dot<DotUnroll>(aic, aiend, a_value, bic, b_value, c_tmp);
		c_value(*my_cic, c_tmp + dot_tmp);
	    }
	private:
	    a_value_type const&   a_value;
	    b_value_type const&   b_value;
	    c_value_type&         c_value;
	    a_cur_type const&     ac;
	    b_cur_type const&     bc;
	    c_icur_type const&    cic;
	};
	

       void mult_add_fast_outer(MatrixA const& a, MatrixB const& b, MatrixC& c)
        {
	    if (a.num_rows() % OuterUnroll != 0)
		throw "B's number of rows must be divisible by MiddleUnroll";

	    a_value_type   a_value(a);
	    b_value_type   b_value(b);
	    c_value_type   c_value(c);
    		
            a_cur_type ac= begin<row_t>(a), aend= end<row_t>(a);
            for (c_cur_type cc= begin<row_t>(c); ac != aend; ac+= OuterUnroll, cc+= OuterUnroll) {
		middle_block my_middle_block(a_value, b_value, c_value, ac, b, cc);
		multi_action_block<middle_block, OuterUnroll>() (my_middle_block);
	    }
        }

	struct middle_block
	{
	    middle_block(a_value_type const& a_value, b_value_type const& b_value, c_value_type& c_value, 
			 a_cur_type const& ac, MatrixB const& b, c_cur_type const& cc) 
		: a_value(a_value), b_value(b_value), c_value(c_value), ac(ac), b(b), cc(cc)
	    {}

	    void operator()(unsigned step)
	    {
		// std::cout << "In middle_block: step " << step << "\n";
		a_cur_type my_ac= ac + step;
		c_cur_type my_cc= cc + step;
		
		b_cur_type bc= begin<col_t>(b), bend= end<col_t>(b);
		for (c_icur_type cic= begin<all_t>(my_cc); bc != bend; bc+= MiddleUnroll, cic+= MiddleUnroll) { 

		    fast_inner_block my_inner_block(a_value, b_value, c_value, my_ac, bc, cic);
		    multi_action_block<fast_inner_block, MiddleUnroll>() (my_inner_block);
		}
	    }
	private:
	    a_value_type const&   a_value;
	    b_value_type const&   b_value;
	    c_value_type&         c_value;
	    a_cur_type const&     ac;
	    MatrixB const&        b;
	    c_cur_type const&     cc;
	};
	
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


/*

Unrolling matrix product with dimensions that are not multiples of blocks

1. Do with optimization:
   C_nw += A_nw * B_nw
   - wherby the matrix dimensions of sub-matrices are the largest multiples of block sizes 
     smaller or equal to the matrix dimensions of the original matrix


2. Do without optimization
   C_nw += A_ne * B_sw
   C_ne += A_n * B_e
   C_s += A_s * B

The inner loop can be unrolled arbitrarily. So, we can simplify

1. Do with optimization:
   C_nw += A_n * B_w
   - wherby the matrix dimensions of sub-matrices are the largest multiples of block sizes 
     smaller or equal to the matrix dimensions of the original matrix


2. Do with optimization only in inner loop
   C_ne += A_n * B_e
   C_s += A_s * B
  

*/
    template <typename MatrixA, typename MatrixB, typename MatrixC, 
	      unsigned DotUnroll= 8, unsigned MiddleUnroll= 4, unsigned OuterUnroll= 2>
    struct mult_add_fast_outer_t
    {
	void operator() (MatrixA const& a, MatrixB const& b, MatrixC& c)
	{
	    matrix_mult_variations<MatrixA, MatrixB, MatrixC, DotUnroll, MiddleUnroll, OuterUnroll> object;

	    typename MatrixC::size_type m= c.num_rows(), m_blocked= (m/OuterUnroll) * OuterUnroll,
	      k= a.num_cols(), k_blocked= (k/MiddleUnroll) * MiddleUnroll,
	      c_row_split= c.begin_row() + m_blocked, c_col_split= c.begin_col() + k_blocked;
	    typename MatrixA::size_type a_row_split= a.begin_row() + m_blocked;
	    typename MatrixB::size_type b_col_split= b.begin_col() + k_blocked;

	    MatrixA a_n= sub_matrix(a, a.begin_row(), a_row_split, a.begin_col(), a.end_col()),
	      a_s= sub_matrix(a, a_row_split, a.end_row(), a.begin_col(), a.end_col());
	    MatrixB b_w= sub_matrix(b, b.begin_row(), b.end_row(), b.begin_col(), b_col_split),
	      b_e= sub_matrix(b, b.begin_row(), b.end_row(), b_col_split, b.end_col());

	    MatrixC c_nw= sub_matrix(c, c.begin_row(), c_row_split, c.begin_col(), c_col_split),
	      c_ne= sub_matrix(c, c.begin_row(), c_row_split, c_col_split, c.end_col()),
	      c_s= sub_matrix(c, c_row_split, c.end_row(), c.begin_col(), c.end_col());

	    object.mult_add_fast_outer(a_n, b_w, c_nw);
	    object.mult_add_fast_dot(a_n, b_e, c_ne);
	    object.mult_add_fast_dot(a_s, b, c_s);

	    // object.mult_add_fast_outer(a, b, c);
	}
    };

    template <typename MatrixA, typename MatrixB, typename MatrixC, unsigned DotUnroll= 8, unsigned MiddleUnroll= 4>
    struct mult_add_fast_middle_t
    {
	void operator() (MatrixA const& a, MatrixB const& b, MatrixC& c)
	{
	    mult_add_fast_outer_t<MatrixA, MatrixB, MatrixC, DotUnroll, MiddleUnroll, 1> fast_outer; 
	    fast_outer(a, b, c);
#if 0
	    // Has less overhead if loops can be unrolled perfectly, otherwise crashes
	    matrix_mult_variations<MatrixA, MatrixB, MatrixC, DotUnroll, MiddleUnroll> object;
	    object.mult_add_fast_middle(a, b, c);
#endif
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
    mult_add_simple(a, b, c);
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
    mult_add_fast_dot(a, b, c);
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void mult_add_fast_middle(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    functor::mult_add_fast_middle_t<MatrixA, MatrixB, MatrixC, 8, 4>()(a, b, c);
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void matrix_mult_fast_middle(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    set_to_0(c);
    mult_add_fast_middle(a, b, c);
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void mult_add_fast_outer(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    functor::mult_add_fast_outer_t<MatrixA, MatrixB, MatrixC, 8, 4, 2>()(a, b, c);
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void matrix_mult_fast_outer(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    set_to_0(c);
    mult_add_fast_outer(a, b, c);
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void matrix_mult(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    std::cout << "matrix_mult without parameters\n";
    set_to_0(c);
    functor::mult_add_fast_outer_t<MatrixA, MatrixB, MatrixC, 8, 4, 2>()(a, b, c);
}

template <unsigned DotUnroll, typename MatrixA, typename MatrixB, typename MatrixC>
void matrix_mult(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    std::cout << "matrix_mult with 1 parameter\n";
    set_to_0(c);
    functor::mult_add_fast_outer_t<MatrixA, MatrixB, MatrixC, DotUnroll, 4, 2>()(a, b, c);
}

template <unsigned DotUnroll, unsigned MiddleUnroll, 
	  typename MatrixA, typename MatrixB, typename MatrixC>
void matrix_mult(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    std::cout << "matrix_mult with 2 parameters\n";
    set_to_0(c);
    functor::mult_add_fast_outer_t<MatrixA, MatrixB, MatrixC, DotUnroll, MiddleUnroll, 2>()(a, b, c);
}

template <unsigned DotUnroll, unsigned MiddleUnroll, unsigned OuterUnroll, 
	  typename MatrixA, typename MatrixB, typename MatrixC>
void matrix_mult(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    std::cout << "matrix_mult with 3 parameters\n";
    set_to_0(c);
    functor::mult_add_fast_outer_t<MatrixA, MatrixB, MatrixC, DotUnroll, MiddleUnroll, OuterUnroll>()(a, b, c);
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
void check_hessian_matrix_product(Matrix const& c, typename Matrix::size_type reduced_dim)
{
    typedef typename Matrix::value_type    value_type;
    typedef typename Matrix::size_type     size_type;
    size_type  rb= c.begin_row(), rl= c.end_row() - 1,
               cb= c.begin_col(), cl= c.end_col() - 1;

    if (!similar_values(value_type(hessian_product_i_j(rb, cb, reduced_dim)), c[rb][cb])) {
	std::cout << "Result in c[" << rb << "][" << cb << "] should be " << hessian_product_i_j(rb, cb, reduced_dim)
	     << " but is " << c[rb][cb] << "\n";
	throw "Wrong result"; }

    if (!similar_values(value_type(hessian_product_i_j(rl, cb, reduced_dim)), c[rl][cb])) {
	std::cout << "Result in c[" << rl << "][" << cb << "] should be " << hessian_product_i_j(rl, cb, reduced_dim)
	     << " but is " << c[rl][cb] << "\n";
	throw "Wrong result"; }

    if (!similar_values(value_type(hessian_product_i_j(rb, cl, reduced_dim)), c[rb][cl])) {
	std::cout << "Result in c[" << rb << "][" << cb << "] should be " << hessian_product_i_j(rb, cl, reduced_dim)
	     << " but is " << c[rb][cl] << "\n";
	throw "Wrong result"; }

    if (!similar_values(value_type(hessian_product_i_j(rl, cl, reduced_dim)), c[rl][cl])) {
	std::cout << "Result in c[" << rl << "][" << cb << "] should be " << hessian_product_i_j(rl, cl, reduced_dim)
	     << " but is " << c[rl][cl] << "\n";
	throw "Wrong result"; }

    // In the center of the matrix
    if (!similar_values(value_type(hessian_product_i_j((rb+rl)/2, (cb+cl)/2, reduced_dim)), c[(rb+rl)/2][(cb+cl)/2])) {
	std::cout << "Result in c[" << (rb+rl)/2 << "][" << (cb+cl)/2 << "] should be " 
		  << hessian_product_i_j((rb+rl)/2, (cb+cl)/2, reduced_dim)
	     << " but is " << c[(rb+rl)/2][(cb+cl)/2] << "\n";
	throw "Wrong result"; }
}








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

    matrix_mult_fast_dot(da, db, dc);
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




