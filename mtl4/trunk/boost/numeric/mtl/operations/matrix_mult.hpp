// $COPYRIGHT$

#ifndef MTL_MATRIX_MULT_INCLUDE
#define MTL_MATRIX_MULT_INCLUDE

#include <boost/numeric/mtl/operations/set_to_0.hpp>
#include <boost/numeric/mtl/range_generator.hpp>
#include <boost/numeric/mtl/operations/cursor_pseudo_dot.hpp>
#include <boost/numeric/mtl/operations/multi_action_block.hpp>

#include <boost/numeric/mtl/dense2D.hpp>



// Define defaults if not yet given as Compiler flag
#ifndef MTL_MATRIX_MULT_OUTER_UNROLL
#  define MTL_MATRIX_MULT_OUTER_UNROLL 1
#endif

#ifndef MTL_MATRIX_MULT_MIDDLE_UNROLL
#  define MTL_MATRIX_MULT_MIDDLE_UNROLL 4
#endif

#ifndef MTL_MATRIX_MULT_INNER_UNROLL
#  define MTL_MATRIX_MULT_INNER_UNROLL 8
#endif


namespace mtl {

namespace functor {

    template <typename MatrixA, typename MatrixB, typename MatrixC, 
	      unsigned InnerUnroll= MTL_MATRIX_MULT_INNER_UNROLL, 
	      unsigned MiddleUnroll= MTL_MATRIX_MULT_MIDDLE_UNROLL, 
	      unsigned OuterUnroll= MTL_MATRIX_MULT_OUTER_UNROLL>
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

	// template<unsigned InnerUnroll>
        void mult_add_fast_inner(MatrixA const& a, MatrixB const& b, MatrixC& c)
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
			dot_tmp= cursor_pseudo_dot<InnerUnroll>(aic, aiend, a_value, bic, b_value, c_tmp);
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
		    dot_tmp= cursor_pseudo_dot<InnerUnroll>(aic, aiend, a_value, bic, b_value, c_tmp);
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

    template <typename MatrixA, typename MatrixB, typename MatrixC, 
	      unsigned InnerUnroll= MTL_MATRIX_MULT_INNER_UNROLL>
    struct mult_add_fast_inner_t
    {
	void operator() (MatrixA const& a, MatrixB const& b, MatrixC& c) const
	{
	    matrix_mult_variations<MatrixA, MatrixB, MatrixC, InnerUnroll> object;
	    object.mult_add_fast_inner(a, b, c);
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
	      unsigned InnerUnroll= MTL_MATRIX_MULT_INNER_UNROLL, 
	      unsigned MiddleUnroll= MTL_MATRIX_MULT_MIDDLE_UNROLL, 
	      unsigned OuterUnroll= MTL_MATRIX_MULT_OUTER_UNROLL>
    struct mult_add_fast_outer_t
    {
	void operator() (MatrixA const& a, MatrixB const& b, MatrixC& c) const
	{
	    matrix_mult_variations<MatrixA, MatrixB, MatrixC, InnerUnroll, MiddleUnroll, OuterUnroll> object;

	    typename MatrixC::size_type m= c.num_rows(), m_blocked= (m/OuterUnroll) * OuterUnroll,
	      k= c.num_cols(), k_blocked= (k/MiddleUnroll) * MiddleUnroll,
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
	    object.mult_add_fast_inner(a_n, b_e, c_ne);
	    object.mult_add_fast_inner(a_s, b, c_s);

	    // object.mult_add_fast_outer(a, b, c);
	}
    };

    template <typename MatrixA, typename MatrixB, typename MatrixC, 
	      unsigned InnerUnroll= MTL_MATRIX_MULT_INNER_UNROLL, 
	      unsigned MiddleUnroll= MTL_MATRIX_MULT_MIDDLE_UNROLL>
    struct mult_add_fast_middle_t
    {
	void operator() (MatrixA const& a, MatrixB const& b, MatrixC& c) const
	{
	    mult_add_fast_outer_t<MatrixA, MatrixB, MatrixC, InnerUnroll, MiddleUnroll> fast_outer; 
	    fast_outer(a, b, c);
#if 0
	    // Has less overhead if loops can be unrolled perfectly, otherwise crashes
	    matrix_mult_variations<MatrixA, MatrixB, MatrixC, InnerUnroll, MiddleUnroll> object;
	    object.mult_add_fast_middle(a, b, c);
#endif
	}
    };


    struct mult_add_empty_t
    {
	template <typename MatrixA, typename MatrixB, typename MatrixC>
	void operator() (MatrixA const& a, MatrixB const& b, MatrixC& c) const
	{
	}
    };


    // fast version for 32x32 double
    struct mult_add_row_times_col_major_32_t
    {
	typedef dense2D<double, matrix_parameters<col_major> > b_type;

	void operator() (dense2D<double> const& a, dense2D<double, matrix_parameters<col_major> > const& b,
			 dense2D<double> c)
	{
	    if (a.num_rows() != 32 || a.num_cols() != 32 || b.num_cols() != 32) {
		mult_add_fast_outer_t<dense2D<double>, dense2D<double, matrix_parameters<col_major> >, dense2D<double> >()(a, b, c);
		return;
	    }
	    dense2D<double>& a_nc= const_cast<dense2D<double>&>(a);
	    b_type& b_nc= const_cast<b_type&>(b);
	    
	    

	    for (unsigned i= 0; i < 32; i++)
		for  (unsigned k= 0; k < 32; k+= 2) {
		    double tmp00= 0.0, tmp01= 0.0, tmp02= 0.0, tmp03= 0.0;
		    for (const double *ap= &a_nc(i, 0), *aend= &a_nc(i, 32), *bp= &b_nc(0, k); ap != aend; ap+= 4, bp+= 4) {
			tmp00+= *ap * *bp;
 			tmp01+= *(ap+1) * *(bp+1);
			tmp02+= *(ap+2) * *(bp+2);
			tmp03+= *(ap+3) * *(bp+3);
		    }
		    c[i][k]+= (tmp00 + tmp01 + tmp02 + tmp03);
		    //#if 0
		    double tmp10= 0.0, tmp11= 0.0, tmp12= 0.0, tmp13= 0.0;
		    for (const double *ap= &a_nc(i, 0), *aend= &a_nc(i, 32), *bp= &b_nc(0, k+1); ap != aend; ap+= 4, bp+= 4) {
			tmp10+= *ap * *bp;
			tmp11+= *(ap+1) * *(bp+1);
			tmp12+= *(ap+2) * *(bp+2);
			tmp13+= *(ap+3) * *(bp+3);
		    }
		    c[i][k+1]+= tmp10 + tmp11 + tmp12 + tmp13;
		    //#endif
		}
	}
    };

#if 0
    // fast version for 32x32 double
    struct mult_add_row_times_col_major_32_cast_t
    {
	typedef dense2D<double, matrix_parameters<col_major> > b_type;

      bool ff(double a[][32]) {
	return true;
      }

	void operator() (dense2D<double> const& a, b_type const& b,
			 dense2D<double> c)
	{
	    if (a.num_rows() != 32 || a.num_cols() != 32 || b.num_cols() != 32) {
		mult_add_fast_outer_t<dense2D<double>, dense2D<double, matrix_parameters<col_major> >, dense2D<double> >()(a, b, c);
		return;
	    }
	    
	    dense2D<double>& a_nc= const_cast<dense2D<double>&>(a), &c_nc= const_cast<dense2D<double>&>(c);
	    b_type& b_nc= const_cast<b_type&>(b);
	    
	    //double *aa[32]= &a_nc(0, 0), *ba[32]= &b_nc(0, 0), *ca[32]= &c_nc(0, 0);
	    double *ap= &a_nc(0, 0);
	    (double *)aa[32]; //, *ba[32], *ca[32];
	    ff(ap);

 	    aa= reinterpret_cast<double(*)[32]> (ap); 
// 	    ca= &c_nc(0, 0); ca= &c_nc(0, 0); 

	    for (unsigned i= 0; i < 32; i++)
		for  (unsigned k= 0; k < 32; k++) {
		    double tmp00= 0.0, tmp01= 0.0, tmp02= 0.0, tmp03= 0.0;

		    // ba is c-array (row-major) but b is column-major -> access like transposed matrix, i.e. row <-> col
		    for (const double *ap= &aa[i][0], *aend= &aa[i][32], *bp= &ba[k][0]; ap != aend; ap+= 4, bp+= 4) {
			tmp00+= *ap * *bp;
			tmp01+= *(ap+1) * *(bp+1);
			tmp02+= *(ap+2) * *(bp+2);
			tmp03+= *(ap+3) * *(bp+3);
		    }
		    ca[i][k]+= (tmp00 + tmp01 + tmp02 + tmp03);
#if 0
		    double tmp10= 0.0, tmp11= 0.0, tmp12= 0.0, tmp13= 0.0;
		    for (const double *ap= &a_nc(i, 0), *aend= &a_nc(i, 32), *bp= &b_nc(0, k+1); ap != aend; ap+= 4, bp+= 4) {
			tmp10+= *ap * *bp;
			tmp11+= *(ap+1) * *(bp+1);
			tmp12+= *(ap+2) * *(bp+2);
			tmp13+= *(ap+3) * *(bp+3);
		    }
		    c[i][k+1]+= tmp10 + tmp11 + tmp12 + tmp13;
#endif
		}
	}
    };
#endif


    // fast version for 16x16 double
    struct mult_add_row_times_col_major_16_t
    {
	typedef dense2D<double, matrix_parameters<col_major> > b_type;

	void operator() (dense2D<double> const& a, dense2D<double, matrix_parameters<col_major> > const& b,
			 dense2D<double> c)
	{
	    if (a.num_rows() != 16 || a.num_cols() != 16 || b.num_cols() != 16) {
		mult_add_fast_outer_t<dense2D<double>, dense2D<double, matrix_parameters<col_major> >, dense2D<double> >()(a, b, c);
		return;
	    }
	    dense2D<double>& a_nc= const_cast<dense2D<double>&>(a);
	    b_type& b_nc= const_cast<b_type&>(b);
	    
	    for (unsigned i= 0; i < 16; i++)
		for  (unsigned k= 0; k < 16; k++) {
		    double tmp00= 0.0, tmp01= 0.0, tmp02= 0.0, tmp03= 0.0;
		    for (const double *ap= &a_nc(i, 0), *aend= &a_nc(i, 16), *bp= &b_nc(0, k); ap != aend; ap+= 4, bp+= 4) {
			tmp00+= *ap * *bp;
			tmp01+= *(ap+1) * *(bp+1);
			tmp02+= *(ap+2) * *(bp+2);
			tmp03+= *(ap+3) * *(bp+3);
		    }
		    c[i][k]+= (tmp00 + tmp01 + tmp02 + tmp03);
#if 0
		    double tmp10= 0.0, tmp11= 0.0, tmp12= 0.0, tmp13= 0.0;
		    for (const double *ap= &a_nc(i, 0), *aend= &a_nc(i, 16), *bp= &b_nc(0, k+1); ap != aend; ap+= 4, bp+= 4) {
			tmp10+= *ap * *bp;
			tmp11+= *(ap+1) * *(bp+1);
			tmp12+= *(ap+2) * *(bp+2);
			tmp13+= *(ap+3) * *(bp+3);
		    }
		    c[i][k+1]+= tmp10 + tmp11 + tmp12 + tmp13;
#endif
		}
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
void mult_add_fast_inner(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    functor::mult_add_fast_inner_t<MatrixA, MatrixB, MatrixC, 8>()(a, b, c);
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void matrix_mult_fast_inner(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    set_to_0(c);
    mult_add_fast_inner(a, b, c);
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
    functor::mult_add_fast_outer_t<MatrixA, MatrixB, MatrixC>()(a, b, c);
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
    functor::mult_add_fast_outer_t<MatrixA, MatrixB, MatrixC>()(a, b, c);
}

template <unsigned InnerUnroll, typename MatrixA, typename MatrixB, typename MatrixC>
void matrix_mult(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    std::cout << "matrix_mult with 1 parameter\n";
    set_to_0(c);
    functor::mult_add_fast_outer_t<MatrixA, MatrixB, MatrixC, InnerUnroll>()(a, b, c);
}

template <unsigned InnerUnroll, unsigned MiddleUnroll, 
	  typename MatrixA, typename MatrixB, typename MatrixC>
void matrix_mult(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    std::cout << "matrix_mult with 2 parameters\n";
    set_to_0(c);
    functor::mult_add_fast_outer_t<MatrixA, MatrixB, MatrixC, InnerUnroll, MiddleUnroll>()(a, b, c);
}

template <unsigned InnerUnroll, unsigned MiddleUnroll, unsigned OuterUnroll, 
	  typename MatrixA, typename MatrixB, typename MatrixC>
void matrix_mult(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    std::cout << "matrix_mult with 3 parameters\n";
    set_to_0(c);
    functor::mult_add_fast_outer_t<MatrixA, MatrixB, MatrixC, InnerUnroll, MiddleUnroll, OuterUnroll>()(a, b, c);
}


#define MTL_UGLY_DENSE_MAT_MAT_MULT_ITERATOR_TYPEDEFS \
	using glas::tags::row_t; using glas::tags::col_t; using glas::tags::all_t;           \
	using glas::tags::all_cit; using glas::tags::all_it; using traits::range_generator;  \
        typedef typename range_generator<row_t, MatrixA>::type       a_cur_type;             \
        typedef typename range_generator<row_t, MatrixC>::type       c_cur_type;             \
	typedef typename range_generator<col_t, MatrixB>::type       b_cur_type;             \
        typedef typename range_generator<all_it, c_cur_type>::type   c_icur_type;            \
        typedef typename range_generator<all_cit, a_cur_type>::type  a_icur_type;            \
        typedef typename range_generator<all_cit, b_cur_type>::type  b_icur_type;          


template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign>
struct gen_dense_mat_mat_mult_t
{
    void operator()(MatrixA const& a, MatrixB const& b, MatrixC& c)
    {
	// MTL_UGLY_DENSE_MAT_MAT_MULT_ITERATOR_TYPEDEFS
	using glas::tags::row_t; using glas::tags::col_t; using glas::tags::all_t;           
	using glas::tags::all_cit; using glas::tags::all_it; using traits::range_generator;  
        typedef typename range_generator<row_t, MatrixA>::type       a_cur_type;             
        typedef typename range_generator<row_t, MatrixC>::type       c_cur_type;             
	typedef typename range_generator<col_t, MatrixB>::type       b_cur_type;             
        typedef typename range_generator<all_it, c_cur_type>::type   c_icur_type;            
        typedef typename range_generator<all_cit, a_cur_type>::type  a_icur_type;            
        typedef typename range_generator<all_cit, b_cur_type>::type  b_icur_type;          

	if (Assign::init_to_0) set_to_0(c);

	a_cur_type ac= begin<row_t>(a), aend= end<row_t>(a);
	for (c_cur_type cc= begin<row_t>(c); ac != aend; ++ac, ++cc) {

	    b_cur_type bc= begin<col_t>(b), bend= end<col_t>(b);
	    for (c_icur_type cic= begin<all_it>(cc); bc != bend; ++bc, ++cic) { 
		    
		typename MatrixC::value_type c_tmp(*cic);
		a_icur_type aic= begin<all_cit>(ac), aiend= end<all_cit>(ac); 
		for (b_icur_type bic= begin<all_cit>(bc); aic != aiend; ++aic, ++bic)
		    Assign::update(c_tmp, a_value(*aic) * b_value(*bic));
		
		*cic= c_tmp;
	    }
	}
    }    
};

template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign, 
	  typename Backup= gen_dense_mat_mat_mult_t<MatrixA, MatrixB, MatrixC, Assign> >
struct gen_blas_dense_mat_mat_mult_t
    : public Backup
{};


#ifdef MTL_HAS_BLAS 

template<typename ParaA, typename ParaB, typename ParaC, typename Backup>
struct gen_blas_dense_mat_mat_mult_t<dense2D<float, ParaA>, dense2D<float, ParaB>, 
				     dense2D<float, ParaC>, modes::mult_assign_t, Backup>
    : public Backup
{
    void operator(const dense2D<float, ParaA>& a, const dense2D<float, ParaB>& b, 
		  dense2D<float, ParaC>& c)
    {
	int atrans= boost::is_same<typename ParaA::orientation, col_major>::value, ... ;
	fgemm(, atrans, &a[0][0], ...)
    }
};


#endif


namespace functor {

    using glas::tags::row_t; using glas::tags::col_t; using glas::tags::all_t;
    using glas::tags::all_cit; using glas::tags::all_it; using traits::range_generator;

    template <typename MatrixA, typename MatrixB, typename MatrixC, 
	      typename Assign, 
	      unsigned InnerUnroll= MTL_MATRIX_MULT_INNER_UNROLL, 
	      unsigned MiddleUnroll= MTL_MATRIX_MULT_MIDDLE_UNROLL, 
	      unsigned OuterUnroll= MTL_MATRIX_MULT_OUTER_UNROLL>>
    struct gen_matrix_mult_variations
    {
        typedef typename range_generator<row_t, MatrixA>::type     a_cur_type;
        typedef typename range_generator<row_t, MatrixC>::type     c_cur_type;
	typedef typename range_generator<col_t, MatrixB>::type     b_cur_type;
        typedef typename range_generator<all_t, c_cur_type>::type  c_icur_type;
        typedef typename range_generator<all_t, a_cur_type>::type  a_icur_type;
        typedef typename range_generator<all_t, b_cur_type>::type  b_icur_type;

	void gen_dense_mat_mat_mult(MatrixA const& a, MatrixB const& b, MatrixC& c)
	{
        typedef typename range_generator<row_t, MatrixA>::type     a_cur_type;
        typedef typename range_generator<row_t, MatrixC>::type     c_cur_type;
	typedef typename range_generator<col_t, MatrixB>::type     b_cur_type;
        typedef typename range_generator<all_t, c_cur_type>::type  c_icur_type;
        typedef typename range_generator<all_t, a_cur_type>::type  a_icur_type;
        typedef typename range_generator<all_t, b_cur_type>::type  b_icur_type;

	    if (Assign::init_to_0) set_to_0(c);

            a_cur_type ac= begin<row_t>(a), aend= end<row_t>(a);
            for (c_cur_type cc= begin<row_t>(c); ac != aend; ++ac, ++cc) {

		b_cur_type bc= begin<col_t>(b), bend= end<col_t>(b);
		for (c_icur_type cic= begin<all_it>(cc); bc != bend; ++bc, ++cic) { 
		    
		    typename MatrixC::value_type c_tmp(*cic);
		    a_icur_type aic= begin<all_cit>(ac), aiend= end<all_cit>(ac); 
		    for (b_icur_type bic= begin<all_cit>(bc); aic != aiend; ++aic, ++bic)
			Assign::update(c_tmp, a_value(*aic) * b_value(*bic));

		    *cic= c_tmp;
		}
	    }
	}

    };

} // namespace functor






} // namespace mtl

#endif // MTL_MATRIX_MULT_INCLUDE
