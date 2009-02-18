// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MULT_ASSIGN_MODE_INCLUDE
#define MTL_MULT_ASSIGN_MODE_INCLUDE

#include <boost/numeric/mtl/operation/assign_mode.hpp>
#include <boost/numeric/mtl/operation/dmat_dmat_mult.hpp>

namespace mtl { namespace assign {

namespace detail {

    template <typename Assign>
    struct subm_assign
    {
	typedef Assign type;
    };

    template<> 
    struct subm_assign<assign_sum>
    {
	typedef plus_sum type;
    };

}

// Set assign_mode of functor 'Mult' to 'Assign'
// including assign_mode of backup functors and functors for sub-matrices
template <typename Mult, typename Assign>
struct mult_assign_mode 
{};


#if 0
// Omit the fully typed functors; they shouldn't be used directly
template <typename MatrixA, typename MatrixB, typename MatrixC, typename OldAssign, typename Backup,
	  typename Assign> 
struct mult_assign_mode<gen_dmat_dmat_mult_ft<MatrixA, MatrixB, MatrixC, OldAssign, Backup>,
			Assign>
{
    typedef gen_dmat_dmat_mult_ft<MatrixA, MatrixB, MatrixC, Assign, Backup> type;
};
#endif


template <typename OldAssign, typename Backup, typename Assign> 
struct mult_assign_mode<gen_dmat_dmat_mult_t<OldAssign, Backup>, Assign>
{
    typedef gen_dmat_dmat_mult_t<Assign, Backup> type;
};

template <typename OldAssign, typename Backup, typename Assign> 
struct mult_assign_mode<gen_cursor_dmat_dmat_mult_t<OldAssign, Backup>, Assign>
{
    typedef gen_cursor_dmat_dmat_mult_t<Assign, Backup> type;
};

template <unsigned long Tiling1, unsigned long Tiling2, typename OldAssign, typename Backup, typename Assign> 
struct mult_assign_mode<gen_tiling_dmat_dmat_mult_t<Tiling1, Tiling2, OldAssign, Backup>, Assign> 
{
    typedef typename mult_assign_mode<Backup, Assign>::type                      backup_type;
    typedef gen_tiling_dmat_dmat_mult_t<Tiling1, Tiling2, Assign, backup_type>   type;
};

template <typename OldAssign, typename Backup, typename Assign> 
struct mult_assign_mode<gen_tiling_44_dmat_dmat_mult_t<OldAssign, Backup>, Assign> 
{
    typedef typename mult_assign_mode<Backup, Assign>::type                      backup_type;
    typedef gen_tiling_44_dmat_dmat_mult_t<Assign, backup_type>                  type;
};

template <typename OldAssign, typename Backup, typename Assign> 
struct mult_assign_mode<gen_tiling_22_dmat_dmat_mult_t<OldAssign, Backup>, Assign> 
{
    typedef typename mult_assign_mode<Backup, Assign>::type                      backup_type;
    typedef gen_tiling_22_dmat_dmat_mult_t<Assign, backup_type>                  type;
};

template <typename BaseMult, typename BaseTest, typename OldAssign, typename Backup, typename Assign> 
struct mult_assign_mode<gen_recursive_dmat_dmat_mult_t<BaseMult, BaseTest, OldAssign, Backup>, Assign> 
{
    typedef typename mult_assign_mode<Backup, Assign>::type                      backup_type;

    // Corresponding assignment type for sub-matrices
    typedef typename detail::subm_assign<Assign>::type                           base_assign_type;
    typedef typename mult_assign_mode<BaseMult, base_assign_type>::type          base_mult_type;

    typedef gen_recursive_dmat_dmat_mult_t<base_mult_type, BaseTest, Assign, backup_type>  type;
};

template <typename OldAssign, typename Backup, typename Assign> 
struct mult_assign_mode<gen_platform_dmat_dmat_mult_t<OldAssign, Backup>, Assign> 
{
    typedef typename mult_assign_mode<Backup, Assign>::type                      backup_type;
    typedef gen_platform_dmat_dmat_mult_t<Assign, backup_type>                   type;
};

template <typename OldAssign, typename Backup, typename Assign> 
struct mult_assign_mode<gen_blas_dmat_dmat_mult_t<OldAssign, Backup>, Assign> 
{
    typedef typename mult_assign_mode<Backup, Assign>::type                      backup_type;
    typedef gen_blas_dmat_dmat_mult_t<Assign, backup_type>                       type;
};

}} // namespace mtl::assign

#endif // MTL_MULT_ASSIGN_MODE_INCLUDE
