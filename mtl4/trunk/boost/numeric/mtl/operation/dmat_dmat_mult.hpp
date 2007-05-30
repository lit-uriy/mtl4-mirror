// $COPYRIGHT$

#ifndef MTL_DMAT_DMAT_MULT_INCLUDE
#define MTL_DMAT_DMAT_MULT_INCLUDE

#include <boost/type_traits.hpp>

#include <boost/numeric/mtl/operation/set_to_zero.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>
#include <boost/numeric/mtl/operation/cursor_pseudo_dot.hpp>
#include <boost/numeric/mtl/operation/multi_action_block.hpp>
#include <boost/numeric/mtl/operation/assign_mode.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/glas_tag.hpp>
#include <boost/numeric/mtl/utility/is_row_major.hpp>
#include <boost/numeric/meta_math/loop.hpp>
#include <boost/numeric/mtl/recursion/base_case_test.hpp>
#include <boost/numeric/mtl/recursion/base_case_matrix.hpp>
#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>
#include <boost/numeric/mtl/recursion/base_case_cast.hpp>
#include <boost/numeric/mtl/interface/blas.hpp>

#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/operation/print_matrix.hpp>
#include <boost/numeric/mtl/operation/no_op.hpp>

#include <iostream>

namespace mtl {

// =====================================
// Generic matrix product with iterators
// =====================================

template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign= assign::assign_sum,
	  typename Backup= no_op>     // To allow 5th parameter, is ignored
struct gen_dmat_dmat_mult_ft
{
    void operator()(MatrixA const& a, MatrixB const& b, MatrixC& c)
    {
	using namespace tag;
	using traits::range_generator;  
        typedef typename range_generator<row, MatrixA>::type       a_cur_type;             
        typedef typename range_generator<row, MatrixC>::type       c_cur_type;             
	typedef typename range_generator<col, MatrixB>::type       b_cur_type;             
        typedef typename range_generator<iter::all, c_cur_type>::type   c_icur_type;            
        typedef typename range_generator<const_iter::all, a_cur_type>::type  a_icur_type;            
        typedef typename range_generator<const_iter::all, b_cur_type>::type  b_icur_type;          

	if (Assign::init_to_zero) set_to_zero(c);

	a_cur_type ac= begin<row>(a), aend= end<row>(a);
	for (c_cur_type cc= begin<row>(c); ac != aend; ++ac, ++cc) {

	    b_cur_type bc= begin<col>(b), bend= end<col>(b);
	    for (c_icur_type cic= begin<iter::all>(cc); bc != bend; ++bc, ++cic) { 
		    
		typename MatrixC::value_type c_tmp(*cic);
		a_icur_type aic= begin<const_iter::all>(ac), aiend= end<const_iter::all>(ac); 
		for (b_icur_type bic= begin<const_iter::all>(bc); aic != aiend; ++aic, ++bic) {
		    //std::cout << "aic " << *aic << ", bic " << *bic << '\n'; std::cout.flush();
		    Assign::update(c_tmp, *aic * *bic);
		}
		*cic= c_tmp;
	    }
	}
    }    
};


template <typename Assign= assign::assign_sum,
	  typename Backup= no_op>     // To allow 2nd parameter, is ignored
struct gen_dmat_dmat_mult_t
{
    template <typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(MatrixA const& a, MatrixB const& b, MatrixC& c)
    {
	gen_dmat_dmat_mult_ft<MatrixA, MatrixB, MatrixC, Assign, Backup>()(a, b, c);
    }
};


// =====================================================
// Generic matrix product with cursors and property maps
// =====================================================


template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign= assign::assign_sum,
	  typename Backup= no_op>     // To allow 5th parameter, is ignored
struct gen_cursor_dmat_dmat_mult_ft
{
    void operator()(MatrixA const& a, MatrixB const& b, MatrixC& c)
    {
	typedef glas::tag::row                                          row;
	typedef glas::tag::col                                          col;
	typedef glas::tag::all                                          all;

        typedef typename traits::const_value<MatrixA>::type                a_value_type;
        typedef typename traits::const_value<MatrixB>::type                b_value_type;
        typedef typename traits::value<MatrixC>::type                      c_value_type;

        typedef typename traits::range_generator<row, MatrixA>::type     a_cur_type;
        typedef typename traits::range_generator<row, MatrixC>::type     c_cur_type;
        
        typedef typename traits::range_generator<col, MatrixB>::type     b_cur_type;
        typedef typename traits::range_generator<all, c_cur_type>::type  c_icur_type;

        typedef typename traits::range_generator<all, a_cur_type>::type  a_icur_type;
        typedef typename traits::range_generator<all, b_cur_type>::type  b_icur_type;

	if (Assign::init_to_zero) set_to_zero(c);

	a_value_type   a_value(a);
	b_value_type   b_value(b);
	c_value_type   c_value(c);
    		
	a_cur_type ac= begin<row>(a), aend= end<row>(a);
	for (c_cur_type cc= begin<row>(c); ac != aend; ++ac, ++cc) {
	    
	    b_cur_type bc= begin<col>(b), bend= end<col>(b);
	    for (c_icur_type cic= begin<all>(cc); bc != bend; ++bc, ++cic) { 
		
		typename MatrixC::value_type c_tmp(c_value(*cic));
		a_icur_type aic= begin<all>(ac), aiend= end<all>(ac); 
		for (b_icur_type bic= begin<all>(bc); aic != aiend; ++aic, ++bic)
		    Assign::update(c_tmp, a_value(*aic) * b_value(*bic));
		c_value(*cic, c_tmp);
	    }
	} 
    }
};


template <typename Assign= assign::assign_sum,
	  typename Backup= no_op>     // To allow 2nd parameter, is ignored
struct gen_cursor_dmat_dmat_mult_t
{
    template <typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(MatrixA const& a, MatrixB const& b, MatrixC& c)
    {
	gen_cursor_dmat_dmat_mult_ft<MatrixA, MatrixB, MatrixC, Assign, Backup>()(a, b, c);
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

// =======================
// Unrolled with iterators
// required has_2D_layout
// =======================

// Define defaults if not yet given as Compiler flag
#ifndef MTL_DMAT_DMAT_MULT_TILING1
#  define MTL_DMAT_DMAT_MULT_TILING1 2
#endif

#ifndef MTL_DMAT_DMAT_MULT_TILING2
#  define MTL_DMAT_DMAT_MULT_TILING2 4
#endif

#ifndef MTL_DMAT_DMAT_MULT_INNER_UNROLL
#  define MTL_DMAT_DMAT_MULT_INNER_UNROLL 8
#endif


template <unsigned long Index0, unsigned long Max0, unsigned long Index1, unsigned long Max1, typename Assign>
struct gen_tiling_dmat_dmat_mult_block
    : public meta_math::loop2<Index0, Max0, Index1, Max1>
{
    typedef meta_math::loop2<Index0, Max0, Index1, Max1>                              base;
    typedef gen_tiling_dmat_dmat_mult_block<base::next_index0, Max0, base::next_index1, Max1, Assign>  next_t;

    template <typename Value, typename ValueA, typename SizeA, typename ValueB, typename SizeB>
    static inline void apply(Value& tmp00, Value& tmp01, Value& tmp02, Value& tmp03, Value& tmp04, 
			     Value& tmp05, Value& tmp06, Value& tmp07, Value& tmp08, Value& tmp09, 
			     Value& tmp10, Value& tmp11, Value& tmp12, Value& tmp13, Value& tmp14, Value& tmp15, 
			     ValueA *begin_a, SizeA& ari, ValueB *begin_b, SizeB& bci)
    {
	tmp00+= begin_a[ base::index0 * ari ] * begin_b[ base::index1 * bci ];
	next_t::apply(tmp01, tmp02, tmp03, tmp04, tmp05, tmp06, tmp07, tmp08, tmp09, 
		      tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp00, 
		      begin_a, ari, begin_b, bci); 
    }

    template <typename Value, typename MatrixC, typename SizeC>
    static inline void update(Value& tmp00, Value& tmp01, Value& tmp02, Value& tmp03, Value& tmp04, 
			      Value& tmp05, Value& tmp06, Value& tmp07, Value& tmp08, Value& tmp09, 
			      Value& tmp10, Value& tmp11, Value& tmp12, Value& tmp13, Value& tmp14, Value& tmp15,
			      MatrixC& c, SizeC i, SizeC k)
    {
	Assign::update(c(i + base::index0, k + base::index1), tmp00);
	next_t::update(tmp01, tmp02, tmp03, tmp04, tmp05, tmp06, tmp07, tmp08, tmp09, 
		       tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp00, 
		       c, i, k);
    }
};

template <unsigned long Max0, unsigned long Max1, typename Assign>
struct gen_tiling_dmat_dmat_mult_block<Max0, Max0, Max1, Max1, Assign>
    : public meta_math::loop2<Max0, Max0, Max1, Max1>
{
    typedef meta_math::loop2<Max0, Max0, Max1, Max1>  base;

    template <typename Value, typename ValueA, typename SizeA, typename ValueB, typename SizeB>
    static inline void apply(Value& tmp00, Value& tmp01, Value& tmp02, Value& tmp03, Value& tmp04, 
			     Value& tmp05, Value& tmp06, Value& tmp07, Value& tmp08, Value& tmp09, 
			     Value& tmp10, Value& tmp11, Value& tmp12, Value& tmp13, Value& tmp14, Value& tmp15, 
			     ValueA *begin_a, SizeA& ari, ValueB *begin_b, SizeB& bci)
    {
	tmp00+= begin_a[ base::index0 * ari ] * begin_b[ base::index1 * bci ];
    }

    template <typename Value, typename MatrixC, typename SizeC>
    static inline void update(Value& tmp00, Value& tmp01, Value& tmp02, Value& tmp03, Value& tmp04, 
			      Value& tmp05, Value& tmp06, Value& tmp07, Value& tmp08, Value& tmp09, 
			      Value& tmp10, Value& tmp11, Value& tmp12, Value& tmp13, Value& tmp14, Value& tmp15,
			      MatrixC& c, SizeC i, SizeC k)
    {
	Assign::update(c(i + base::index0, k + base::index1), tmp00);
    }
};


template <typename MatrixA, typename MatrixB, typename MatrixC,
	  unsigned long Tiling1= MTL_DMAT_DMAT_MULT_TILING1,
	  unsigned long Tiling2= MTL_DMAT_DMAT_MULT_TILING2,
	  typename Assign= assign::assign_sum, 
	  typename Backup= gen_dmat_dmat_mult_t<Assign> >
struct gen_tiling_dmat_dmat_mult_ft
{
    BOOST_STATIC_ASSERT(Tiling1 * Tiling2 <= 16);
  
    void operator()(MatrixA const& a, MatrixB const& b, MatrixC& c)
    {
	apply(a, b, c, typename traits::category<MatrixA>::type(),
	      typename traits::category<MatrixB>::type());
    }   
 
private:
    void apply(MatrixA const& a, MatrixB const& b, MatrixC& c, tag::universe, tag::universe)
    {
	Backup()(a, b, c);
    }

#if MTL_OUTLINE_TILING_DMAT_DMAT_MULT_APPLY
    void apply(MatrixA const& a, MatrixB const& b, MatrixC& c, tag::has_2D_layout, tag::has_2D_layout);
#else
    void apply(MatrixA const& a, MatrixB const& b, MatrixC& c, tag::has_2D_layout, tag::has_2D_layout)
    {
	// std::cout << "do unrolling\n";

	if (Assign::init_to_zero) set_to_zero(c);

	typedef gen_tiling_dmat_dmat_mult_block<1, Tiling1, 1, Tiling2, Assign>  block;
	typedef typename MatrixC::size_type                                          size_type;
	typedef typename MatrixC::value_type                                         value_type;
	const value_type z= math::zero(c[0][0]);    // if this are matrices we need their size

	size_type i_max= c.num_rows(), i_block= Tiling1 * (i_max / Tiling1),
	          k_max= c.num_cols(), k_block= Tiling2 * (k_max / Tiling2);
	size_t ari= &a(1, 0) - &a(0, 0), // how much is the offset of A's entry increased by incrementing row
	       aci= &a(0, 1) - &a(0, 0), bri= &b(1, 0) - &b(0, 0), bci= &b(0, 1) - &b(0, 0);
	    
	// C_nw += A_nw * B_nw
	for (size_type i= 0; i < i_block; i+= Tiling1)
	    for (size_type k= 0; k < k_block; k+= Tiling2) {

		value_type tmp00= z, tmp01= z, tmp02= z, tmp03= z, tmp04= z,
                           tmp05= z, tmp06= z, tmp07= z, tmp08= z, tmp09= z,
 		           tmp10= z, tmp11= z, tmp12= z, tmp13= z, tmp14= z, tmp15= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri)
		    block::apply(tmp00, tmp01, tmp02, tmp03, tmp04, tmp05, tmp06, tmp07, tmp08, tmp09, 
				 tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, 
				 begin_a, ari, begin_b, bci); 
		block::update(tmp00, tmp01, tmp02, tmp03, tmp04, tmp05, tmp06, tmp07, tmp08, tmp09, 
			      tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, 
			      c, i, k);
	    }

	// C_ne += A_n * B_e
	for (size_type i= 0; i < i_block; i++)
	    for (int k = k_block; k < k_max; k++) {
		value_type tmp00= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri)
		    tmp00 += *begin_a * *begin_b;
		Assign::update(c(i, k), tmp00);
	    }

	// C_s += A_s * B
	for (size_type i= i_block; i < i_max; i++)
	    for (int k = 0; k < k_max; k++) {
		value_type tmp00= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri)
		    tmp00 += *begin_a * *begin_b;
		Assign::update(c(i, k), tmp00);
	    }
    }
#endif
};

template <unsigned long Tiling1= MTL_DMAT_DMAT_MULT_TILING1,
	  unsigned long Tiling2= MTL_DMAT_DMAT_MULT_TILING2,
	  typename Assign= assign::assign_sum, 
	  typename Backup= gen_dmat_dmat_mult_t<Assign> >
struct gen_tiling_dmat_dmat_mult_t
{
    template <typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(MatrixA const& a, MatrixB const& b, MatrixC& c)
    {
	gen_tiling_dmat_dmat_mult_ft<
	     MatrixA, MatrixB, MatrixC, Tiling1, Tiling2, Assign, Backup
	>()(a, b, c);
    }
};


#if MTL_OUTLINE_TILING_DMAT_DMAT_MULT_APPLY
template <typename MatrixA, typename MatrixB, typename MatrixC, 
	  unsigned long Tiling1, unsigned long Tiling2,
	  typename Assign, typename Backup>
void gen_tiling_dmat_dmat_mult_ft<MatrixA, MatrixB, MatrixC, Tiling1, Tiling2, Assign, Backup>::
apply(MatrixA const& a, MatrixB const& b, MatrixC& c, tag::has_2D_layout, tag::has_2D_layout)
{
	// std::cout << "do unrolling\n";

	if (Assign::init_to_zero) set_to_zero(c);

	typedef gen_tiling_dmat_dmat_mult_block<1, Tiling1, 1, Tiling2, Assign>  block;
	typedef typename MatrixC::size_type                                          size_type;
	typedef typename MatrixC::value_type                                         value_type;
	const value_type z= math::zero(c[0][0]);    // if this are matrices we need their size

	size_type i_max= c.num_rows(), i_block= Tiling1 * (i_max / Tiling1),
	          k_max= c.num_cols(), k_block= Tiling2 * (k_max / Tiling2);
	size_t ari= &a(1, 0) - &a(0, 0), // how much is the offset of A's entry increased by incrementing row
	       aci= &a(0, 1) - &a(0, 0), bri= &b(1, 0) - &b(0, 0), bci= &b(0, 1) - &b(0, 0);
	    
	// C_nw += A_nw * B_nw
	for (size_type i= 0; i < i_block; i+= Tiling1)
	    for (size_type k= 0; k < k_block; k+= Tiling2) {

		value_type tmp00= z, tmp01= z, tmp02= z, tmp03= z, tmp04= z,
                           tmp05= z, tmp06= z, tmp07= z, tmp08= z, tmp09= z,
 		           tmp10= z, tmp11= z, tmp12= z, tmp13= z, tmp14= z, tmp15= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri)
		    block::apply(tmp00, tmp01, tmp02, tmp03, tmp04, tmp05, tmp06, tmp07, tmp08, tmp09, 
				 tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, 
				 begin_a, ari, begin_b, bci); 
		block::update(tmp00, tmp01, tmp02, tmp03, tmp04, tmp05, tmp06, tmp07, tmp08, tmp09, 
			      tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, 
			      c, i, k);
	    }

	// C_ne += A_n * B_e
	for (size_type i= 0; i < i_block; i++)
	    for (int k = k_block; k < k_max; k++) {
		value_type tmp00= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri)
		    tmp00 += *begin_a * *begin_b;
		Assign::update(c(i, k), tmp00);
	    }

	// C_s += A_s * B
	for (size_type i= i_block; i < i_max; i++)
	    for (int k = 0; k < k_max; k++) {
		value_type tmp00= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri)
		    tmp00 += *begin_a * *begin_b;
		Assign::update(c(i, k), tmp00);
	    }
}
#endif

// =================================
// Unrolled with iterators fixed 4x4
// required has_2D_layout
// =================================


template <typename MatrixA, typename MatrixB, typename MatrixC, 
	  typename Assign= assign::assign_sum, 
	  typename Backup= gen_dmat_dmat_mult_t<Assign> >
struct gen_tiling_44_dmat_dmat_mult_ft
{
    void operator()(MatrixA const& a, MatrixB const& b, MatrixC& c)
    {
	apply(a, b, c, typename traits::category<MatrixA>::type(),
	      typename traits::category<MatrixB>::type());
    }   
 
private:
    void apply(MatrixA const& a, MatrixB const& b, MatrixC& c, tag::universe, tag::universe)
    {
	Backup()(a, b, c);
    }

#if MTL_OUTLINE_TILING_DMAT_DMAT_MULT_APPLY
    void apply(MatrixA const& a, MatrixB const& b, MatrixC& c, tag::has_2D_layout, tag::has_2D_layout);
#else
    void apply(MatrixA const& a, MatrixB const& b, MatrixC& c, tag::has_2D_layout, tag::has_2D_layout)
    {
        // std::cout << "do unrolling\n";

	if (Assign::init_to_zero) set_to_zero(c);

	typedef typename MatrixC::size_type                                          size_type;
	typedef typename MatrixC::value_type                                         value_type;

	const size_type  Tiling1= 4, Tiling2= 4;
	const value_type z= math::zero(c[0][0]);    // if this are matrices we need their size

	size_type i_max= c.num_rows(), i_block= Tiling1 * (i_max / Tiling1),
	          k_max= c.num_cols(), k_block= Tiling2 * (k_max / Tiling2);
	size_t ari= &a(1, 0) - &a(0, 0), // how much is the offset of A's entry increased by incrementing row
	       aci= &a(0, 1) - &a(0, 0), bri= &b(1, 0) - &b(0, 0), bci= &b(0, 1) - &b(0, 0);

	// C_nw += A_nw * B_nw
	for (size_type i= 0; i < i_block; i+= Tiling1)
	    for (size_type k= 0; k < k_block; k+= Tiling2) {

		value_type tmp00= z, tmp01= z, tmp02= z, tmp03= z, tmp04= z,
                           tmp05= z, tmp06= z, tmp07= z, tmp08= z, tmp09= z,
 		           tmp10= z, tmp11= z, tmp12= z, tmp13= z, tmp14= z, tmp15= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri) {
		    tmp00+= begin_a[ 0 * ari ] * begin_b[ 0 * bci ];
		    tmp01+= begin_a[ 0 * ari ] * begin_b[ 1 * bci ];
		    tmp02+= begin_a[ 0 * ari ] * begin_b[ 2 * bci ];
		    tmp03+= begin_a[ 0 * ari ] * begin_b[ 3 * bci ];
		    tmp04+= begin_a[ 1 * ari ] * begin_b[ 0 * bci ];
		    tmp05+= begin_a[ 1 * ari ] * begin_b[ 1 * bci ];
		    tmp06+= begin_a[ 1 * ari ] * begin_b[ 2 * bci ];
		    tmp07+= begin_a[ 1 * ari ] * begin_b[ 3 * bci ];
		    tmp08+= begin_a[ 2 * ari ] * begin_b[ 0 * bci ];
		    tmp09+= begin_a[ 2 * ari ] * begin_b[ 1 * bci ];
		    tmp10+= begin_a[ 2 * ari ] * begin_b[ 2 * bci ];
		    tmp11+= begin_a[ 2 * ari ] * begin_b[ 3 * bci ];
		    tmp12+= begin_a[ 3 * ari ] * begin_b[ 0 * bci ];
		    tmp13+= begin_a[ 3 * ari ] * begin_b[ 1 * bci ];
		    tmp14+= begin_a[ 3 * ari ] * begin_b[ 2 * bci ];
		    tmp15+= begin_a[ 3 * ari ] * begin_b[ 3 * bci ];
		}
		Assign::update(c(i + 0, k + 0), tmp00);
		Assign::update(c(i + 0, k + 1), tmp01);
		Assign::update(c(i + 0, k + 2), tmp02);
		Assign::update(c(i + 0, k + 3), tmp03);
		Assign::update(c(i + 1, k + 0), tmp04);
		Assign::update(c(i + 1, k + 1), tmp05);
		Assign::update(c(i + 1, k + 2), tmp06);
		Assign::update(c(i + 1, k + 3), tmp07);
		Assign::update(c(i + 2, k + 0), tmp08);
		Assign::update(c(i + 2, k + 1), tmp09);
		Assign::update(c(i + 2, k + 2), tmp10);
		Assign::update(c(i + 2, k + 3), tmp11);
		Assign::update(c(i + 3, k + 0), tmp12);
		Assign::update(c(i + 3, k + 1), tmp13);
		Assign::update(c(i + 3, k + 2), tmp14);
		Assign::update(c(i + 3, k + 3), tmp15);
	    }

	// C_ne += A_n * B_e
	for (size_type i= 0; i < i_block; i++)
	    for (int k = k_block; k < k_max; k++) {
		value_type tmp00= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri)
		    tmp00 += *begin_a * *begin_b;
		Assign::update(c(i, k), tmp00);
	    }

	// C_s += A_s * B
	for (size_type i= i_block; i < i_max; i++)
	    for (int k = 0; k < k_max; k++) {
		value_type tmp00= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri)
		    tmp00 += *begin_a * *begin_b;
		Assign::update(c(i, k), tmp00);
	    }
    }
#endif
};

template <typename Assign= assign::assign_sum, 
	  typename Backup= gen_dmat_dmat_mult_t<Assign> >
struct gen_tiling_44_dmat_dmat_mult_t
{
    template <typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(MatrixA const& a, MatrixB const& b, MatrixC& c)
    {
	gen_tiling_44_dmat_dmat_mult_ft<
	     MatrixA, MatrixB, MatrixC, Assign, Backup
	>()(a, b, c);
    }
};


#if MTL_OUTLINE_TILING_DMAT_DMAT_MULT_APPLY
template <typename MatrixA, typename MatrixB, typename MatrixC, 
	  typename Assign, typename Backup>
void gen_tiling_44_dmat_dmat_mult_ft<MatrixA, MatrixB, MatrixC, Assign, Backup>::
apply(MatrixA const& a, MatrixB const& b, MatrixC& c, tag::has_2D_layout, tag::has_2D_layout)
{
        // std::cout << "do unrolling\n";

	if (Assign::init_to_zero) set_to_zero(c);

	typedef typename MatrixC::size_type                                          size_type;
	typedef typename MatrixC::value_type                                         value_type;

	const size_type  Tiling1= 4, Tiling2= 4;
	const value_type z= math::zero(c[0][0]);    // if this are matrices we need their size


	size_type i_max= c.num_rows(), i_block= Tiling1 * (i_max / Tiling1),
	          k_max= c.num_cols(), k_block= Tiling2 * (k_max / Tiling2);
	size_t ari= &a(1, 0) - &a(0, 0), // how much is the offset of A's entry increased by incrementing row
	       aci= &a(0, 1) - &a(0, 0), bri= &b(1, 0) - &b(0, 0), bci= &b(0, 1) - &b(0, 0);

	// C_nw += A_nw * B_nw
	for (size_type i= 0; i < i_block; i+= Tiling1)
	    for (size_type k= 0; k < k_block; k+= Tiling2) {

		value_type tmp00= z, tmp01= z, tmp02= z, tmp03= z, tmp04= z,
                           tmp05= z, tmp06= z, tmp07= z, tmp08= z, tmp09= z,
 		           tmp10= z, tmp11= z, tmp12= z, tmp13= z, tmp14= z, tmp15= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri) {
		    tmp00+= begin_a[ 0 * ari ] * begin_b[ 0 * bci ];
		    tmp01+= begin_a[ 0 * ari ] * begin_b[ 1 * bci ];
		    tmp02+= begin_a[ 0 * ari ] * begin_b[ 2 * bci ];
		    tmp03+= begin_a[ 0 * ari ] * begin_b[ 3 * bci ];
		    tmp04+= begin_a[ 1 * ari ] * begin_b[ 0 * bci ];
		    tmp05+= begin_a[ 1 * ari ] * begin_b[ 1 * bci ];
		    tmp06+= begin_a[ 1 * ari ] * begin_b[ 2 * bci ];
		    tmp07+= begin_a[ 1 * ari ] * begin_b[ 3 * bci ];
		    tmp08+= begin_a[ 2 * ari ] * begin_b[ 0 * bci ];
		    tmp09+= begin_a[ 2 * ari ] * begin_b[ 1 * bci ];
		    tmp10+= begin_a[ 2 * ari ] * begin_b[ 2 * bci ];
		    tmp11+= begin_a[ 2 * ari ] * begin_b[ 3 * bci ];
		    tmp12+= begin_a[ 3 * ari ] * begin_b[ 0 * bci ];
		    tmp13+= begin_a[ 3 * ari ] * begin_b[ 1 * bci ];
		    tmp14+= begin_a[ 3 * ari ] * begin_b[ 2 * bci ];
		    tmp15+= begin_a[ 3 * ari ] * begin_b[ 3 * bci ];
		}
		Assign::update(c(i + 0, k + 0), tmp00);
		Assign::update(c(i + 0, k + 1), tmp01);
		Assign::update(c(i + 0, k + 2), tmp02);
		Assign::update(c(i + 0, k + 3), tmp03);
		Assign::update(c(i + 1, k + 0), tmp04);
		Assign::update(c(i + 1, k + 1), tmp05);
		Assign::update(c(i + 1, k + 2), tmp06);
		Assign::update(c(i + 1, k + 3), tmp07);
		Assign::update(c(i + 2, k + 0), tmp08);
		Assign::update(c(i + 2, k + 1), tmp09);
		Assign::update(c(i + 2, k + 2), tmp10);
		Assign::update(c(i + 2, k + 3), tmp11);
		Assign::update(c(i + 3, k + 0), tmp12);
		Assign::update(c(i + 3, k + 1), tmp13);
		Assign::update(c(i + 3, k + 2), tmp14);
		Assign::update(c(i + 3, k + 3), tmp15);
	    }

	// C_ne += A_n * B_e
	for (size_type i= 0; i < i_block; i++)
	    for (int k = k_block; k < k_max; k++) {
		value_type tmp00= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri)
		    tmp00 += *begin_a * *begin_b;
		Assign::update(c(i, k), tmp00);
	    }

	// C_s += A_s * B
	for (size_type i= i_block; i < i_max; i++)
	    for (int k = 0; k < k_max; k++) {
		value_type tmp00= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri)
		    tmp00 += *begin_a * *begin_b;
		Assign::update(c(i, k), tmp00);
	    }
}
#endif



// =================================
// Unrolled with iterators fixed 2x2
// required has_2D_layout
// =================================


template <typename MatrixA, typename MatrixB, typename MatrixC, 
	  typename Assign= assign::assign_sum, 
	  typename Backup= gen_dmat_dmat_mult_t<Assign> >
struct gen_tiling_22_dmat_dmat_mult_ft
{
    void operator()(MatrixA const& a, MatrixB const& b, MatrixC& c)
    {
	apply(a, b, c, typename traits::category<MatrixA>::type(),
	      typename traits::category<MatrixB>::type());
    }   
 
private:
    void apply(MatrixA const& a, MatrixB const& b, MatrixC& c, tag::universe, tag::universe)
    {
	Backup()(a, b, c);
    }

#if MTL_OUTLINE_TILING_DMAT_DMAT_MULT_APPLY
    void apply(MatrixA const& a, MatrixB const& b, MatrixC& c, tag::has_2D_layout, tag::has_2D_layout);
#else
    void apply(MatrixA const& a, MatrixB const& b, MatrixC& c, tag::has_2D_layout, tag::has_2D_layout)
    {
        // std::cout << "do unrolling\n";

	if (Assign::init_to_zero) set_to_zero(c);

	typedef typename MatrixC::size_type                                          size_type;
	typedef typename MatrixC::value_type                                         value_type;

	const size_type  Tiling1= 2, Tiling2= 2;
	const value_type z= math::zero(c[0][0]);    // if this are matrices we need their size

	size_type i_max= c.num_rows(), i_block= Tiling1 * (i_max / Tiling1),
	          k_max= c.num_cols(), k_block= Tiling2 * (k_max / Tiling2);
	size_t ari= &a(1, 0) - &a(0, 0), // how much is the offset of A's entry increased by incrementing row
	       aci= &a(0, 1) - &a(0, 0), bri= &b(1, 0) - &b(0, 0), bci= &b(0, 1) - &b(0, 0);

	// C_nw += A_nw * B_nw
	for (size_type i= 0; i < i_block; i+= Tiling1)
	    for (size_type k= 0; k < k_block; k+= Tiling2) {

		value_type tmp00= z, tmp01= z, tmp02= z, tmp03= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri) {
		    tmp00+= begin_a[ 0 ] * begin_b[ 0 ];
		    tmp01+= begin_a[ 0 ] * begin_b[bci];
		    tmp02+= begin_a[ari] * begin_b[ 0 ];
		    tmp03+= begin_a[ari] * begin_b[bci];
		}
		Assign::update(c(i + 0, k + 0), tmp00);
		Assign::update(c(i + 0, k + 1), tmp01);
		Assign::update(c(i + 1, k + 0), tmp02);
		Assign::update(c(i + 1, k + 1), tmp03);
	    }

	// C_ne += A_n * B_e
	for (size_type i= 0; i < i_block; i++)
	    for (int k = k_block; k < k_max; k++) {
		value_type tmp00= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri)
		    tmp00 += *begin_a * *begin_b;
		Assign::update(c(i, k), tmp00);
	    }

	// C_s += A_s * B
	for (size_type i= i_block; i < i_max; i++)
	    for (int k = 0; k < k_max; k++) {
		value_type tmp00= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri)
		    tmp00 += *begin_a * *begin_b;
		Assign::update(c(i, k), tmp00);
	    }
    }
#endif
};

template <typename Assign= assign::assign_sum, 
	  typename Backup= gen_dmat_dmat_mult_t<Assign> >
struct gen_tiling_22_dmat_dmat_mult_t
{
    template <typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(MatrixA const& a, MatrixB const& b, MatrixC& c)
    {
	gen_tiling_22_dmat_dmat_mult_ft<
	     MatrixA, MatrixB, MatrixC, Assign, Backup
	>()(a, b, c);
    }
};


#if MTL_OUTLINE_TILING_DMAT_DMAT_MULT_APPLY
template <typename MatrixA, typename MatrixB, typename MatrixC, 
	  typename Assign, typename Backup>
void gen_tiling_22_dmat_dmat_mult_ft<MatrixA, MatrixB, MatrixC, Assign, Backup>::
apply(MatrixA const& a, MatrixB const& b, MatrixC& c, tag::has_2D_layout, tag::has_2D_layout)
{
        // std::cout << "do unrolling\n";

	if (Assign::init_to_zero) set_to_zero(c);

	typedef typename MatrixC::size_type                                          size_type;
	typedef typename MatrixC::value_type                                         value_type;

	const size_type  Tiling1= 2, Tiling2= 2;
	const value_type z= math::zero(c[0][0]);    // if this are matrices we need their size

	size_type i_max= c.num_rows(), i_block= Tiling1 * (i_max / Tiling1),
	          k_max= c.num_cols(), k_block= Tiling2 * (k_max / Tiling2);
	size_t ari= &a(1, 0) - &a(0, 0), // how much is the offset of A's entry increased by incrementing row
	       aci= &a(0, 1) - &a(0, 0), bri= &b(1, 0) - &b(0, 0), bci= &b(0, 1) - &b(0, 0);

	// C_nw += A_nw * B_nw
	for (size_type i= 0; i < i_block; i+= Tiling1)
	    for (size_type k= 0; k < k_block; k+= Tiling2) {

		value_type tmp00= z, tmp01= z, tmp02= z, tmp03= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri) {
		    tmp00+= begin_a[ 0 * ari ] * begin_b[ 0 * bci ];
		    tmp01+= begin_a[ 0 * ari ] * begin_b[ 1 * bci ];
		    tmp02+= begin_a[ 1 * ari ] * begin_b[ 0 * bci ];
		    tmp03+= begin_a[ 1 * ari ] * begin_b[ 1 * bci ];
		}
		Assign::update(c(i + 0, k + 0), tmp00);
		Assign::update(c(i + 0, k + 1), tmp01);
		Assign::update(c(i + 1, k + 0), tmp02);
		Assign::update(c(i + 1, k + 1), tmp03);
	    }

	// C_ne += A_n * B_e
	for (size_type i= 0; i < i_block; i++)
	    for (int k = k_block; k < k_max; k++) {
		value_type tmp00= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri)
		    tmp00 += *begin_a * *begin_b;
		Assign::update(c(i, k), tmp00);
	    }

	// C_s += A_s * B
	for (size_type i= i_block; i < i_max; i++)
	    for (int k = 0; k < k_max; k++) {
		value_type tmp00= z;
		const typename MatrixA::value_type *begin_a= &a(i, 0), *end_a= &a(i, a.num_cols());
		const typename MatrixB::value_type *begin_b= &b(0, k);

		for (; begin_a != end_a; begin_a+= aci, begin_b+= bri)
		    tmp00 += *begin_a * *begin_b;
		Assign::update(c(i, k), tmp00);
	    }
}
#endif


// ========================
// Recursive Multiplication
// ========================

namespace wrec {

    template <typename BaseMult, typename BaseTest= recursion::bound_test_static<64> >
    struct gen_dmat_dmat_mult_t
    {
	template <typename RecA, typename RecB, typename RecC>
	void operator()(RecA const& rec_a, RecB const& rec_b, RecC& rec_c)
	{
	    using recursion::base_case_cast;
#if 0
	    std::cout << "\n\n before matrix multiplication:\n";
	    std::cout << "A:\n"; print_matrix_row_cursor(rec_a.get_value());
	    std::cout << "B:\n"; print_matrix_row_cursor(rec_b.get_value());
	    std::cout << "C:\n"; print_matrix_row_cursor(rec_c.get_value());
#endif

	    if (rec_a.is_empty() || rec_b.is_empty() || rec_c.is_empty())
		return;

	    if (BaseTest()(rec_a)) {
		typename recursion::base_case_matrix<typename RecC::matrix_type, BaseTest>::type
		    c= base_case_cast<BaseTest>(rec_c.get_value());
		BaseMult()(base_case_cast<BaseTest>(rec_a.get_value()),
			   base_case_cast<BaseTest>(rec_b.get_value()), c);
	    } else {
		RecC c_north_west= rec_c.north_west(), c_north_east= rec_c.north_east(),
		    c_south_west= rec_c.south_west(), c_south_east= rec_c.south_east();

		(*this)(rec_a.north_west(), rec_b.north_west(), c_north_west);
		(*this)(rec_a.north_west(), rec_b.north_east(), c_north_east);
		(*this)(rec_a.south_west(), rec_b.north_east(), c_south_east);
		(*this)(rec_a.south_west(), rec_b.north_west(), c_south_west);
		(*this)(rec_a.south_east(), rec_b.south_west(), c_south_west);
		(*this)(rec_a.south_east(), rec_b.south_east(), c_south_east);
		(*this)(rec_a.north_east(), rec_b.south_east(), c_north_east);
		(*this)(rec_a.north_east(), rec_b.south_west(), c_north_west);
	    }
#if 0
	    std::cout << "\n\n after matrix multiplication:\n";
	    std::cout << "A:\n"; print_matrix_row_cursor(rec_a.get_value());
	    std::cout << "B:\n"; print_matrix_row_cursor(rec_b.get_value());
	    std::cout << "C:\n"; print_matrix_row_cursor(rec_c.get_value());
#endif
	}
    };

} // namespace wrec


template <typename BaseMult, 
	  typename BaseTest= recursion::bound_test_static<64>,
	  typename Assign= assign::assign_sum, 
	  typename Backup= gen_dmat_dmat_mult_t<Assign> >
struct gen_recursive_dmat_dmat_mult_t
{
    template <typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(MatrixA const& a, MatrixB const& b, MatrixC& c)
    {
	apply(a, b, c, typename traits::category<MatrixA>::type(),
	      typename traits::category<MatrixB>::type(), 
	      typename traits::category<MatrixC>::type());
    }   
 
private:
    // If one matrix is not sub-dividable then take backup function
    template <typename MatrixA, typename MatrixB, typename MatrixC>
    void apply(MatrixA const& a, MatrixB const& b, MatrixC& c, tag::universe, tag::universe, tag::universe)
    {
	Backup()(a, b, c);
    }

    // Only if matrix is sub-dividable, otherwise backup
    template <typename MatrixA, typename MatrixB, typename MatrixC>
    void apply(MatrixA const& a, MatrixB const& b, MatrixC& c, 
	       tag::qsub_dividable, tag::qsub_dividable, tag::qsub_dividable)
    {
	// std::cout << "do recursion\n";

	if (Assign::init_to_zero) set_to_zero(c);

	// Make sure that mult functor of basecase has appropriate assign mode (in all nestings)
	// i.e. replace assign::assign_sum by assign::plus_sum including backup functor
	
	using recursion::matrix_recurator;
	matrix_recurator<MatrixA>    rec_a(a);
	matrix_recurator<MatrixB>    rec_b(b);
	matrix_recurator<MatrixC>    rec_c(c);
	equalize_depth(rec_a, rec_b, rec_c);

	wrec::gen_dmat_dmat_mult_t<BaseMult, BaseTest>() (rec_a, rec_b, rec_c);
    }
};





// ==================================
// Plattform specific implementations
// ==================================

// Here only general definition that calls backup function
// Special implementations needed in other files, which are included at the end

template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign= assign::assign_sum, 
	  typename Backup= gen_dmat_dmat_mult_t<Assign> >
struct gen_platform_dmat_dmat_mult_ft
    : public Backup
{};


template <typename Assign= assign::assign_sum, 
	  typename Backup= gen_dmat_dmat_mult_t<Assign> >
struct gen_platform_dmat_dmat_mult_t
{
    template <typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(MatrixA const& a, MatrixB const& b, MatrixC& c)
    {
	gen_platform_dmat_dmat_mult_ft<MatrixA, MatrixB, MatrixC, Assign, Backup>()(a, b, c);
    }
};


// ==================================
// BLAS functions as far as supported
// ==================================


template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign= assign::assign_sum, 
	  typename Backup= gen_dmat_dmat_mult_t<Assign> >
struct gen_blas_dmat_dmat_mult_ft
    : public Backup
{};


#ifdef MTL_HAS_BLAS

namespace detail {

    // Transform from assign representation to BLAS
    double dgemm_alpha(assign::assign_sum)
    {
	return 1.0;
    }

    double dgemm_alpha(assign::plus_sum)
    {
	return 1.0;
    }
 
    double dgemm_alpha(assign::minus_sum)
    {
	return -1.0;
    }

    // Transform from assign representation to BLAS
    double dgemm_beta(assign::assign_sum)
    {
	return 0.0;
    }

    double dgemm_beta(assign::plus_sum)
    {
	return 1.0;
    }
 
    double dgemm_beta(assign::minus_sum)
    {
	return 1.0;
    }

} // detail
 
// Only sketch
template<typename ParaA, typename ParaB, typename ParaC, typename Backup>
struct gen_blas_dmat_dmat_mult_ft<dense2D<float, ParaA>, dense2D<float, ParaB>, 
				     dense2D<float, ParaC>, assign::assign_sum, Backup>
{
    void operator()(const dense2D<float, ParaA>& a, const dense2D<float, ParaB>& b, 
		    dense2D<float, ParaC>& c)
    {
	std::cout << "pretend BLAS\n";
	Backup()(a, b, c);
#if 0
	int atrans= boost::is_same<typename ParaA::orientation, col_major>::value, ... ;
	fgemm(, atrans, &a[0][0], ...)
#endif
    }
};

template<typename ParaA, typename ParaB, typename ParaC, typename Assign, typename Backup>
struct gen_blas_dmat_dmat_mult_ft<dense2D<double, ParaA>, dense2D<double, ParaB>, 
				     dense2D<double, ParaC>, Assign, Backup>
{
    void operator()(const dense2D<double, ParaA>& a, const dense2D<double, ParaB>& b, 
		    dense2D<double, ParaC>& c)
    {
#if 0
	std::cout << "pretend BLAS\n";
	Backup()(a, b, c);
	
#else
	if (traits::is_row_major<ParaC>::value) {
	    Backup()(a, b, c);
	    return;
	}

	// C needs to be transposed if row-major !!!!!!!!!! That means physically in memory
	std::cout << "use BLAS\n";
	int m= a.num_rows(), n= c.num_cols(), k= a.num_cols(), lda= a.get_ldim(), ldb= b.get_ldim(), ldc= c.get_ldim();
	double alpha= detail::dgemm_alpha(Assign()), beta= detail::dgemm_beta(Assign());

	dgemm_(traits::is_row_major<ParaA>::value ? "T" : "N", traits::is_row_major<ParaB>::value ? "T" : "N",
	       &m, &n, &k, &alpha, &a[0][0], &lda, &b[0][0], &ldb, &beta, &c[0][0], &ldc);
#endif
    }
};

#endif

template <typename Assign= assign::assign_sum, 
	  typename Backup= gen_dmat_dmat_mult_t<Assign> >
struct gen_blas_dmat_dmat_mult_t
    : public Backup
{
    template <typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(MatrixA const& a, MatrixB const& b, MatrixC& c)
    {
	gen_blas_dmat_dmat_mult_ft<MatrixA, MatrixB, MatrixC, Assign, Backup>()(a, b, c);
    }
};



} // namespace mtl

#endif // MTL_DMAT_DMAT_MULT_INCLUDE

// Include plattform specific implementations
#include <boost/numeric/mtl/operation/opteron/matrix_mult.hpp>

