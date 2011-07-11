// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_VPT_VPT_INCLUDE
#define MTL_VPT_VPT_INCLUDE

#ifdef MTL_HAS_VPT
  #include <vt_user.h> 
  #include <string>
  #include <boost/mpl/bool.hpp>
#endif 

namespace mtl { namespace vpt {

#ifdef MTL_HAS_VPT

#ifndef MTL_VPT_LEVEL
#  define MTL_VPT_LEVEL 2
#endif 

template <int N>
class vampir_trace
{
    typedef boost::mpl::bool_<(MTL_VPT_LEVEL * 100 < N)> to_print;
  public:
    vampir_trace() { entry(to_print());  }

    void entry(boost::mpl::false_) {}
    void entry(boost::mpl::true_) 
    {	
	VT_USER_START(name.c_str()); 
	// std::cout << "vpt_entry(" << N << ")\n";    
    }
    
    ~vampir_trace() { end(to_print());  }

    void end(boost::mpl::false_) {}
    void end(boost::mpl::true_) 
    {
	VT_USER_END(name.c_str()); 
	// std::cout << "vpt_end(" << N << ")\n";    
    }
    
    bool is_traced() { return to_print::value; }

  private:
    static std::string name;
};

// Categories:
// Utilities:                       000
// Static size operations:          1000
// Vector operations:               2000
// Matrix Vector & single matrix:   3000
// Matrix matrix operations:        4000
// Factorizations, preconditioners: 5000
// Iterative solvers:               6000


// Utilities:                       000
template <> std::string vampir_trace<0001>::name("copysign");

#if 0
#include <boost/numeric/mtl/interface/vpt.hpp>
    vampir_trace<>0 tracer;

#endif
// Static size operations:          1000
template <> std::string vampir_trace<1001>::name("stat_vec_expr");
template <> std::string vampir_trace<1002>::name("fixed_size_dmat_dmat_mult");

// Vector operations:               2000
template <> std::string vampir_trace<2001>::name("gen_vector_copy");
template <> std::string vampir_trace<2002>::name("cross");
template <> std::string vampir_trace<2003>::name("dot");
template <> std::string vampir_trace<2004>::name("householder");
template <> std::string vampir_trace<2005>::name("householder_s");
template <> std::string vampir_trace<2006>::name("vector::infinity_norm");
template <> std::string vampir_trace<2007>::name("vector::look_at_each_nonzero");
template <> std::string vampir_trace<2008>::name("vector::look_at_each_nonzero_pos");
template <> std::string vampir_trace<2009>::name("vector::reduction");
template <> std::string vampir_trace<2010>::name("vector::max");
template <> std::string vampir_trace<2011>::name("vector::max_abs_pos");
template <> std::string vampir_trace<2012>::name("max_of_sums");
template <> std::string vampir_trace<2013>::name("vector::max_pos");
template <> std::string vampir_trace<2014>::name("merge_complex_vector");
template <> std::string vampir_trace<2015>::name("vector::one_norm");
template <> std::string vampir_trace<2016>::name("vector::diagonal");
template <> std::string vampir_trace<2017>::name("dyn_vec_expr");
template <> std::string vampir_trace<2018>::name("");
template <> std::string vampir_trace<2019>::name("");				      
template <> std::string vampir_trace<2020>::name("");
template <> std::string vampir_trace<2021>::name("");
template <> std::string vampir_trace<2022>::name("");
template <> std::string vampir_trace<2023>::name("");
template <> std::string vampir_trace<2024>::name("");
template <> std::string vampir_trace<2025>::name("");
template <> std::string vampir_trace<2026>::name("");
template <> std::string vampir_trace<2027>::name("");
template <> std::string vampir_trace<2028>::name("");
template <> std::string vampir_trace<2029>::name("");
template <> std::string vampir_trace<2030>::name("");
template <> std::string vampir_trace<2031>::name("");
template <> std::string vampir_trace<2032>::name("");
template <> std::string vampir_trace<2033>::name("");
template <> std::string vampir_trace<2034>::name("");
template <> std::string vampir_trace<2035>::name("");
template <> std::string vampir_trace<2036>::name("");
template <> std::string vampir_trace<2037>::name("");
template <> std::string vampir_trace<2038>::name("");
template <> std::string vampir_trace<2039>::name("");
template <> std::string vampir_trace<2040>::name("");
template <> std::string vampir_trace<2041>::name("");

#if 0
#include <boost/numeric/mtl/interface/vpt.hpp>
    vampir_trace<>0 tracer;

#endif

// Matrix Vector & single matrix:   3000
template <> std::string vampir_trace<3001>::name("matrix_copy_ele_times");
template <> std::string vampir_trace<3002>::name("gen_matrix_copy");
template <> std::string vampir_trace<3003>::name("copy");
template <> std::string vampir_trace<3004>::name("clone");
template <> std::string vampir_trace<3005>::name("compute_summand");
template <> std::string vampir_trace<3006>::name("crop");
template <> std::string vampir_trace<3007>::name("matrix::diagonal");
template <> std::string vampir_trace<3008>::name("assign_each_nonzero");
template <> std::string vampir_trace<3009>::name("fill");
template <> std::string vampir_trace<3010>::name("frobenius_norm");
template <> std::string vampir_trace<3011>::name("matrix::infinity_norm");
template <> std::string vampir_trace<3012>::name("invert_diagonal");
template <> std::string vampir_trace<3013>::name("iota");
template <> std::string vampir_trace<3014>::name("left_scale_inplace");
template <> std::string vampir_trace<3015>::name("matrix::look_at_each_nonzero");
template <> std::string vampir_trace<3016>::name("matrix::look_at_each_nonzero_pos");
template <> std::string vampir_trace<3017>::name("fsize_dense_mat_cvec_mult");
template <> std::string vampir_trace<3018>::name("dense_mat_cvec_mult");
template <> std::string vampir_trace<3019>::name("mvec_cvec_mult");
template <> std::string vampir_trace<3020>::name("trans_mvec_cvec_mult");
template <> std::string vampir_trace<3021>::name("herm_mvec_cvec_mult");
template <> std::string vampir_trace<3022>::name("crs_cvec_mult");
template <> std::string vampir_trace<3023>::name("ccs_cvec_mult");
template <> std::string vampir_trace<3024>::name("matrix::max_abs_pos");
template <> std::string vampir_trace<3025>::name("matrix::one_norm");
template <> std::string vampir_trace<3026>::name("");
template <> std::string vampir_trace<3027>::name("");
template <> std::string vampir_trace<3028>::name("");
template <> std::string vampir_trace<3029>::name("");
template <> std::string vampir_trace<3030>::name("");
template <> std::string vampir_trace<3031>::name("");
template <> std::string vampir_trace<3032>::name("");
template <> std::string vampir_trace<3033>::name("");
template <> std::string vampir_trace<3034>::name("");
template <> std::string vampir_trace<3035>::name("");
template <> std::string vampir_trace<3036>::name("");
template <> std::string vampir_trace<3037>::name("");
template <> std::string vampir_trace<3038>::name("");
template <> std::string vampir_trace<3039>::name("");


#if 0
#include <boost/numeric/mtl/interface/vpt.hpp>
    vampir_trace<>0 tracer;

#endif


// Matrix matrix operations:        4000
template <> std::string vampir_trace<4001>::name("cursor_dmat_dmat_mult");
template <> std::string vampir_trace<4002>::name("dmat_dmat_mult");
template <> std::string vampir_trace<4003>::name("tiling_dmat_dmat_mult");
template <> std::string vampir_trace<4004>::name("tiling_44_dmat_dmat_mult");
template <> std::string vampir_trace<4005>::name("tiling_22_dmat_dmat_mult");
template <> std::string vampir_trace<4006>::name("wrec_dmat_dmat_mult");
template <> std::string vampir_trace<4007>::name("recursive_dmat_dmat_mult");
template <> std::string vampir_trace<4008>::name("xgemm");
template <> std::string vampir_trace<4009>::name("");
template <> std::string vampir_trace<4010>::name("mult");
template <> std::string vampir_trace<4011>::name("gen_mult");
template <> std::string vampir_trace<4012>::name("mat_mat_mult");
template <> std::string vampir_trace<4013>::name("");
template <> std::string vampir_trace<4014>::name("");
template <> std::string vampir_trace<4015>::name("");
template <> std::string vampir_trace<4016>::name("");
template <> std::string vampir_trace<4017>::name("");
template <> std::string vampir_trace<4018>::name("");
template <> std::string vampir_trace<4019>::name("");				      
template <> std::string vampir_trace<4020>::name("");
template <> std::string vampir_trace<4021>::name("");
template <> std::string vampir_trace<4022>::name("");
template <> std::string vampir_trace<4023>::name("");
template <> std::string vampir_trace<4024>::name("");
template <> std::string vampir_trace<4025>::name("");
template <> std::string vampir_trace<4026>::name("");
template <> std::string vampir_trace<4027>::name("");
template <> std::string vampir_trace<4028>::name("");
template <> std::string vampir_trace<4029>::name("");
template <> std::string vampir_trace<4030>::name("");
template <> std::string vampir_trace<4031>::name("");
template <> std::string vampir_trace<4032>::name("");
template <> std::string vampir_trace<4033>::name("");
template <> std::string vampir_trace<4034>::name("");
template <> std::string vampir_trace<4035>::name("");
template <> std::string vampir_trace<4036>::name("");
template <> std::string vampir_trace<4037>::name("");
template <> std::string vampir_trace<4038>::name("");
template <> std::string vampir_trace<4039>::name("");
template <> std::string vampir_trace<4040>::name("");
template <> std::string vampir_trace<4041>::name("");

#if 0
#include <boost/numeric/mtl/interface/vpt.hpp>
    vampir_trace<>0 tracer;

#endif

// Factorizations, preconditioners: 5000
template <> std::string vampir_trace<5001>::name("cholesky_base");
template <> std::string vampir_trace<5002>::name("cholesky_solve_base");
template <> std::string vampir_trace<5003>::name("cholesky_schur_base");
template <> std::string vampir_trace<5004>::name("cholesky_update_base");
template <> std::string vampir_trace<5005>::name("cholesky_schur_update");
template <> std::string vampir_trace<5006>::name("cholesky_tri_solve");
template <> std::string vampir_trace<5007>::name("cholesky_tri_schur");
template <> std::string vampir_trace<5008>::name("recursive cholesky");
template <> std::string vampir_trace<5009>::name("fill_matrix_for_cholesky");
template <> std::string vampir_trace<5010>::name("qr_sym_imp");
template <> std::string vampir_trace<5011>::name("qr_algo");
template <> std::string vampir_trace<5012>::name("eigenvalue_symmetric");
template <> std::string vampir_trace<5013>::name("hessenberg_q");
template <> std::string vampir_trace<5014>::name("hessenberg_factors");
template <> std::string vampir_trace<5015>::name("extract_householder_hessenberg");
template <> std::string vampir_trace<5016>::name("householder_hessenberg");
template <> std::string vampir_trace<5017>::name("extract_hessenberg");
template <> std::string vampir_trace<5018>::name("hessenberg");
template <> std::string vampir_trace<5019>::name("inv_upper");
template <> std::string vampir_trace<5020>::name("inv_lower");
template <> std::string vampir_trace<5021>::name("inv");
template <> std::string vampir_trace<5022>::name("lower_trisolve");
template <> std::string vampir_trace<5023>::name("lu");
template <> std::string vampir_trace<5024>::name("lu(pivot)");
template <> std::string vampir_trace<5025>::name("lu_f");
template <> std::string vampir_trace<5026>::name("lu_solve_straight");
template <> std::string vampir_trace<5027>::name("lu_apply");
template <> std::string vampir_trace<5028>::name("lu_solve");
template <> std::string vampir_trace<5029>::name("lu_adjoint_apply");
template <> std::string vampir_trace<5030>::name("lu_adjoint_solve");
template <> std::string vampir_trace<5031>::name("");
template <> std::string vampir_trace<5032>::name("");
template <> std::string vampir_trace<5033>::name("");
template <> std::string vampir_trace<5034>::name("");
template <> std::string vampir_trace<5035>::name("");
template <> std::string vampir_trace<5036>::name("");
template <> std::string vampir_trace<5037>::name("");
template <> std::string vampir_trace<5038>::name("");
template <> std::string vampir_trace<5039>::name("");
template <> std::string vampir_trace<5040>::name("");
template <> std::string vampir_trace<5041>::name("");
template <> std::string vampir_trace<5042>::name("");
template <> std::string vampir_trace<5043>::name("");
template <> std::string vampir_trace<5044>::name("");
template <> std::string vampir_trace<5045>::name("");
template <> std::string vampir_trace<5046>::name("");
template <> std::string vampir_trace<5047>::name("");
template <> std::string vampir_trace<5048>::name("");
template <> std::string vampir_trace<5049>::name("");

#if 0
#include <boost/numeric/mtl/interface/vpt.hpp>
    vampir_trace<>0 tracer;

#endif

// Iterative solvers:               6000
template <> std::string vampir_trace<6001>::name("cg");






template <> std::string vampir_trace<9999>::name("main");
    

// Only for testing
template <> std::string vampir_trace<1099>::name("helper_function");
template <> std::string vampir_trace<2099>::name("function");


#else
    template <int N>
    class vampir_trace 
    {
      public:
	vampir_trace() {}
	void show_vpt_level() {}
	bool is_traced() { return false; }
    };
#endif


} // namespace mtl

using vpt::vampir_trace;

} // namespace mtl

#endif // MTL_VPT_VPT_INCLUDE