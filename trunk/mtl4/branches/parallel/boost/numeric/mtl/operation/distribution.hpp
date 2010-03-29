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

#ifndef MTL_OPERATION_DISTRIBUTION_INCLUDE
#define MTL_OPERATION_DISTRIBUTION_INCLUDE

#ifdef MTL_HAS_MPI

#include <boost/type_traits/add_reference.hpp>
#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/distribution.hpp>
#include <boost/numeric/mtl/par/distribution.hpp>
#include <boost/numeric/mtl/vector/map_view.hpp>
#include <boost/numeric/mtl/vector/vec_vec_aop_expr.hpp>
#include <boost/numeric/mtl/vector/vec_vec_pmop_expr.hpp>

namespace mtl { 

    namespace vector {

	template <typename Functor, typename Vector>
	typename mtl::traits::distribution<map_view<Functor, Vector> >::type
	inline distribution(const map_view<Functor, Vector>& v)
	{
	    return distribution(v.reference());
	}

	template <typename E1, typename E2, typename Functor>
	typename mtl::traits::distribution<vec_vec_aop_expr<E1, E2, Functor> >::type
	inline distribution(const vec_vec_aop_expr<E1, E2, Functor>& v)
	{
	    MTL_DEBUG_THROW_IF(distribution(v.first) != distribution(v.second), incompatible_distribution());
	    return distribution(v.first);
	}

	template <typename E1, typename E2, typename Functor>
	typename mtl::traits::distribution<vec_vec_pmop_expr<E1, E2, Functor> >::type
	inline distribution(const vec_vec_pmop_expr<E1, E2, Functor>& v)
	{
	    MTL_DEBUG_THROW_IF(distribution(v.reference_first()) != distribution(v.reference_second()), incompatible_distribution());
	    return distribution(v.reference_first());
	}

	template <class Value, typename Parameters>
	inline par::replication distribution(const dense_vector<Value, Parameters>&) { return par::replication(); }
	

    } // namespace vector

    namespace matrix {

        template <typename Value, typename Parameters> 
	inline par::replication row_distribution(const dense2D<Value, Parameters>&) { return par::replication(); }
	
        template <typename Value, typename Parameters> 
	inline par::replication col_distribution(const dense2D<Value, Parameters>&) { return par::replication(); }
	
	template <typename Vector> 
	typename mtl::traits::distribution<Vector>::type
	inline row_distribution(const multi_vector<Vector>& A)
	{
	    typedef typename mtl::traits::distribution<Vector>::type dist_type;
	    //return distribution(A.vector(0)); 
	    return num_cols(A) > 0 ? distribution(A.vector(0)) : dist_type(num_rows(A)); 
	}

	template <typename Vector> 
	inline par::replication col_distribution(const multi_vector<Vector>&) { return par::replication(); }
	
	template <typename Vector> 
	typename mtl::traits::distribution<Vector>::type
	inline row_distribution(const multi_vector_range<Vector>& A) { return row_distribution(A.ref); }

	template <typename Vector> 
	inline par::replication col_distribution(const multi_vector_range<Vector>&) { return par::replication(); }
    }


    template <typename Matrix, typename CVector>
    typename mtl::traits::distribution<mtl::mat_cvec_times_expr<Matrix, CVector> >::type
    inline distribution(const mtl::mat_cvec_times_expr<Matrix, CVector>& expr)
    {
	MTL_DEBUG_THROW_IF(col_distribution(expr.first) != distribution(expr.second), incompatible_distribution());
	return row_distribution(expr.first);
    }

} // namespace mtl

#endif // MTL_HAS_MPI

#endif // MTL_OPERATION_DISTRIBUTION_INCLUDE
