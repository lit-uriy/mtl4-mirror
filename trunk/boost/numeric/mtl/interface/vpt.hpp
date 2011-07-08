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
// Utilities:                       100
// Vector operations:               200
// Matrix Vector & single matrix:   300
// Matrix matrix operations:        400
// Factorizations, preconditioners: 500
// Iterative solvers:               600


// Utilities:                       100
template <> std::string vampir_trace<101>::name("copysign");

#if 0
#include <boost/numeric/mtl/interface/vpt.hpp>
    vampir_trace<> tracer;

#endif

// Vector operations:               200
template <> std::string vampir_trace<201>::name("gen_vector_copy");
template <> std::string vampir_trace<202>::name("cross");
template <> std::string vampir_trace<203>::name("dot");
template <> std::string vampir_trace<204>::name("householder");
template <> std::string vampir_trace<205>::name("householder_s");
template <> std::string vampir_trace<206>::name("vector::infinity_norm");
template <> std::string vampir_trace<207>::name("vector::look_at_each_nonzero");
template <> std::string vampir_trace<208>::name("vector::look_at_each_nonzero_pos");
template <> std::string vampir_trace<209>::name("vector::reduction");
template <> std::string vampir_trace<210>::name("vector::max");
template <> std::string vampir_trace<211>::name("vector::max_abs_pos");
template <> std::string vampir_trace<212>::name("max_of_sums");
template <> std::string vampir_trace<213>::name("vector::max_pos");
template <> std::string vampir_trace<214>::name("merge_complex_vector");
template <> std::string vampir_trace<215>::name("vector::one_norm");
template <> std::string vampir_trace<216>::name("");
template <> std::string vampir_trace<217>::name("");
template <> std::string vampir_trace<218>::name("");
template <> std::string vampir_trace<219>::name("");				      
template <> std::string vampir_trace<220>::name("");
template <> std::string vampir_trace<221>::name("");
template <> std::string vampir_trace<222>::name("");
template <> std::string vampir_trace<223>::name("");
template <> std::string vampir_trace<224>::name("");
template <> std::string vampir_trace<225>::name("");
template <> std::string vampir_trace<226>::name("");
template <> std::string vampir_trace<227>::name("");
template <> std::string vampir_trace<228>::name("");
template <> std::string vampir_trace<229>::name("");
template <> std::string vampir_trace<230>::name("");
template <> std::string vampir_trace<231>::name("");
template <> std::string vampir_trace<232>::name("");
template <> std::string vampir_trace<233>::name("");
template <> std::string vampir_trace<234>::name("");
template <> std::string vampir_trace<235>::name("");
template <> std::string vampir_trace<236>::name("");
template <> std::string vampir_trace<237>::name("");
template <> std::string vampir_trace<238>::name("");
template <> std::string vampir_trace<239>::name("");
template <> std::string vampir_trace<240>::name("");
template <> std::string vampir_trace<241>::name("");

#if 0
#include <boost/numeric/mtl/interface/vpt.hpp>
    vampir_trace<> tracer;

#endif

// Matrix Vector & single matrix:   300
template <> std::string vampir_trace<301>::name("matrix_copy_ele_times");
template <> std::string vampir_trace<302>::name("gen_matrix_copy");
template <> std::string vampir_trace<303>::name("copy");
template <> std::string vampir_trace<304>::name("clone");
template <> std::string vampir_trace<305>::name("compute_summand");
template <> std::string vampir_trace<306>::name("crop");
template <> std::string vampir_trace<307>::name("diagonal");
template <> std::string vampir_trace<308>::name("assign_each_nonzero");
template <> std::string vampir_trace<309>::name("fill");
template <> std::string vampir_trace<310>::name("frobenius_norm");
template <> std::string vampir_trace<311>::name("matrix::infinity_norm");
template <> std::string vampir_trace<312>::name("invert_diagonal");
template <> std::string vampir_trace<313>::name("iota");
template <> std::string vampir_trace<314>::name("left_scale_inplace");
template <> std::string vampir_trace<315>::name("matrix::look_at_each_nonzero");
template <> std::string vampir_trace<316>::name("matrix::look_at_each_nonzero_pos");
template <> std::string vampir_trace<317>::name("fsize_dense_mat_cvec_mult");
template <> std::string vampir_trace<318>::name("dense_mat_cvec_mult");
template <> std::string vampir_trace<319>::name("mvec_cvec_mult");
template <> std::string vampir_trace<320>::name("trans_mvec_cvec_mult");
template <> std::string vampir_trace<321>::name("herm_mvec_cvec_mult");
template <> std::string vampir_trace<322>::name("crs_cvec_mult");
template <> std::string vampir_trace<323>::name("ccs_cvec_mult");
template <> std::string vampir_trace<324>::name("matrix::max_abs_pos");
template <> std::string vampir_trace<325>::name("matrix::one_norm");
template <> std::string vampir_trace<326>::name("");
template <> std::string vampir_trace<327>::name("");
template <> std::string vampir_trace<328>::name("");
template <> std::string vampir_trace<329>::name("");
template <> std::string vampir_trace<330>::name("");
template <> std::string vampir_trace<331>::name("");
template <> std::string vampir_trace<332>::name("");
template <> std::string vampir_trace<333>::name("");
template <> std::string vampir_trace<334>::name("");
template <> std::string vampir_trace<335>::name("");
template <> std::string vampir_trace<336>::name("");
template <> std::string vampir_trace<337>::name("");
template <> std::string vampir_trace<338>::name("");
template <> std::string vampir_trace<339>::name("");


#if 0
#include <boost/numeric/mtl/interface/vpt.hpp>
    vampir_trace<> tracer;

#endif


// Matrix matrix operations:        400
template <> std::string vampir_trace<401>::name("cursor_dmat_dmat_mult");
template <> std::string vampir_trace<402>::name("dmat_dmat_mult");
template <> std::string vampir_trace<403>::name("tiling_dmat_dmat_mult");
template <> std::string vampir_trace<404>::name("tiling_44_dmat_dmat_mult");
template <> std::string vampir_trace<405>::name("tiling_22_dmat_dmat_mult");
template <> std::string vampir_trace<406>::name("wrec_dmat_dmat_mult");
template <> std::string vampir_trace<407>::name("recursive_dmat_dmat_mult");
template <> std::string vampir_trace<408>::name("xgemm");
template <> std::string vampir_trace<409>::name("fixed_size_dmat_dmat_mult");
template <> std::string vampir_trace<410>::name("mult");
template <> std::string vampir_trace<411>::name("gen_mult");
template <> std::string vampir_trace<412>::name("mat_mat_mult");
template <> std::string vampir_trace<413>::name("");
template <> std::string vampir_trace<414>::name("");
template <> std::string vampir_trace<415>::name("");
template <> std::string vampir_trace<416>::name("");
template <> std::string vampir_trace<417>::name("");
template <> std::string vampir_trace<418>::name("");
template <> std::string vampir_trace<419>::name("");				      
template <> std::string vampir_trace<420>::name("");
template <> std::string vampir_trace<421>::name("");
template <> std::string vampir_trace<422>::name("");
template <> std::string vampir_trace<423>::name("");
template <> std::string vampir_trace<424>::name("");
template <> std::string vampir_trace<425>::name("");
template <> std::string vampir_trace<426>::name("");
template <> std::string vampir_trace<427>::name("");
template <> std::string vampir_trace<428>::name("");
template <> std::string vampir_trace<429>::name("");
template <> std::string vampir_trace<430>::name("");
template <> std::string vampir_trace<431>::name("");
template <> std::string vampir_trace<432>::name("");
template <> std::string vampir_trace<433>::name("");
template <> std::string vampir_trace<434>::name("");
template <> std::string vampir_trace<435>::name("");
template <> std::string vampir_trace<436>::name("");
template <> std::string vampir_trace<437>::name("");
template <> std::string vampir_trace<438>::name("");
template <> std::string vampir_trace<439>::name("");
template <> std::string vampir_trace<440>::name("");
template <> std::string vampir_trace<441>::name("");

#if 0
#include <boost/numeric/mtl/interface/vpt.hpp>
    vampir_trace<> tracer;

#endif

// Factorizations, preconditioners: 500
template <> std::string vampir_trace<501>::name("cholesky_base");
template <> std::string vampir_trace<502>::name("cholesky_solve_base");
template <> std::string vampir_trace<503>::name("cholesky_schur_base");
template <> std::string vampir_trace<504>::name("cholesky_update_base");
template <> std::string vampir_trace<505>::name("cholesky_schur_update");
template <> std::string vampir_trace<506>::name("cholesky_tri_solve");
template <> std::string vampir_trace<507>::name("cholesky_tri_schur");
template <> std::string vampir_trace<508>::name("recursive cholesky");
template <> std::string vampir_trace<509>::name("fill_matrix_for_cholesky");
template <> std::string vampir_trace<510>::name("qr_sym_imp");
template <> std::string vampir_trace<511>::name("qr_algo");
template <> std::string vampir_trace<512>::name("eigenvalue_symmetric");
template <> std::string vampir_trace<513>::name("hessenberg_q");
template <> std::string vampir_trace<514>::name("hessenberg_factors");
template <> std::string vampir_trace<515>::name("extract_householder_hessenberg");
template <> std::string vampir_trace<516>::name("householder_hessenberg");
template <> std::string vampir_trace<517>::name("extract_hessenberg");
template <> std::string vampir_trace<518>::name("hessenberg");
template <> std::string vampir_trace<519>::name("inv_upper");
template <> std::string vampir_trace<520>::name("inv_lower");
template <> std::string vampir_trace<521>::name("inv");
template <> std::string vampir_trace<522>::name("lower_trisolve");
template <> std::string vampir_trace<523>::name("lu");
template <> std::string vampir_trace<524>::name("lu(pivot)");
template <> std::string vampir_trace<525>::name("lu_f");
template <> std::string vampir_trace<526>::name("lu_solve_straight");
template <> std::string vampir_trace<527>::name("lu_apply");
template <> std::string vampir_trace<528>::name("lu_solve");
template <> std::string vampir_trace<529>::name("lu_adjoint_apply");
template <> std::string vampir_trace<530>::name("lu_adjoint_solve");
template <> std::string vampir_trace<531>::name("");
template <> std::string vampir_trace<532>::name("");
template <> std::string vampir_trace<533>::name("");
template <> std::string vampir_trace<534>::name("");
template <> std::string vampir_trace<535>::name("");
template <> std::string vampir_trace<536>::name("");
template <> std::string vampir_trace<537>::name("");
template <> std::string vampir_trace<538>::name("");
template <> std::string vampir_trace<539>::name("");
template <> std::string vampir_trace<540>::name("");
template <> std::string vampir_trace<541>::name("");
template <> std::string vampir_trace<542>::name("");
template <> std::string vampir_trace<543>::name("");
template <> std::string vampir_trace<544>::name("");
template <> std::string vampir_trace<545>::name("");
template <> std::string vampir_trace<546>::name("");
template <> std::string vampir_trace<547>::name("");
template <> std::string vampir_trace<548>::name("");
template <> std::string vampir_trace<549>::name("");

#if 0
#include <boost/numeric/mtl/interface/vpt.hpp>
    vampir_trace<> tracer;

#endif

// Iterative solvers:               600
template <> std::string vampir_trace<601>::name("cg");






template <> std::string vampir_trace<999>::name("main");
    

// Only for testing
template <> std::string vampir_trace<199>::name("helper_function");
template <> std::string vampir_trace<299>::name("function");


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
