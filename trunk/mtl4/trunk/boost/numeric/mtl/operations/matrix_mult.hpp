// $COPYRIGHT$

#ifndef MTL_MATRIX_MULT_INCLUDE
#define MTL_MATRIX_MULT_INCLUDE

#include <boost/numeric/mtl/operations/set_to_0.hpp>
#include <boost/numeric/mtl/range_generator.hpp>
#include <boost/numeric/mtl/operations/cursor_pseudo_dot.hpp>
#include <boost/numeric/mtl/operations/multi_action_block.hpp>

namespace mtl {

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
	void operator() (MatrixA const& a, MatrixB const& b, MatrixC& c) const
	{
	    matrix_mult_variations<MatrixA, MatrixB, MatrixC>().mult_add_simple(a, b, c);
	}
    };

    template <typename MatrixA, typename MatrixB, typename MatrixC, unsigned DotUnroll= 8>
    struct mult_add_fast_dot_t
    {
	void operator() (MatrixA const& a, MatrixB const& b, MatrixC& c) const
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
	void operator() (MatrixA const& a, MatrixB const& b, MatrixC& c) const
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
	void operator() (MatrixA const& a, MatrixB const& b, MatrixC& c) const
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




} // namespace mtl

#endif // MTL_MATRIX_MULT_INCLUDE
