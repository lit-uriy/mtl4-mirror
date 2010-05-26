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

#ifndef MTL_VECTOR_DEVICE_EXPR_INCLUDE
#define MTL_VECTOR_DEVICE_EXPR_INCLUDE

#ifdef MTL_HAS_CUDA

#include <boost/numeric/mtl/cuda/vector_cuda.cu>
#include <boost/numeric/mtl/vector/vec_vec_pmop_expr.hpp>
#include <boost/numeric/mtl/vector/map_view.hpp>


namespace mtl { namespace vector {

	template <typename Vector> struct device_expr {}; // must be specialized

	template <typename Value>
	struct device_expr<cuda::vector<Value> >
	{
	    typedef  Value                                                value_type;
	    typedef typename cuda::vector<Value>::size_type               size_type;

	    device_expr(const cuda::vector<Value>& v) : dptr(v.get_device_pointer()) {}

	    __device__ value_type operator[](size_type i) const { return dptr[i]; }
	    
	    const Value *dptr;
	};

	template <typename E1, typename E2, typename SFunctor>
	struct device_expr<vec_vec_pmop_expr<E1, E2, SFunctor> >
	{
	    typedef typename SFunctor::result_type                        value_type;
	    typedef typename E1::size_type                                size_type;

	    device_expr(const vec_vec_pmop_expr<E1, E2, SFunctor>& exp) : first(exp.first.value), second(exp.second.value) {}

	    __device__ value_type operator[](size_type i) const { return SFunctor::apply(first[i], second[i]); }
	    
	    device_expr<E1> first;
	    device_expr<E2> second;
	};

	template <typename Functor, typename Vector> 
	struct device_expr<map_view<Functor, Vector> >
	{
	    typedef typename Functor::result_type              value_type;
	    typedef typename Vector::size_type                 size_type;
	    
	    device_expr(const map_view<Functor, Vector>& exp) : functor(exp.functor), ref(exp.ref) {}

	    __device__ value_type operator[](size_type i) const { return functor(ref[i]); }

	    Functor             functor;
	    device_expr<Vector> ref;
	};

	template <typename Scaling, typename Vector>
	struct device_expr<scaled_view<Scaling, Vector> >
	  : device_expr< map_view<tfunctor::scale<Scaling, typename Vector::value_type>, Vector> >
	{
	    typedef device_expr< map_view<tfunctor::scale<Scaling, typename Vector::value_type>, Vector> > base;
	    device_expr(const scaled_view<Scaling, Vector>& expr) : base(expr) {} 
	};


	template <typename Vector, typename RScaling>
	struct device_expr<rscaled_view<Vector, RScaling> >
	  : device_expr< map_view<tfunctor::rscale<typename Vector::value_type, RScaling>, Vector> >
	{
	    typedef device_expr< map_view<tfunctor::rscale<typename Vector::value_type, RScaling>, Vector> > base;
	    device_expr(const rscaled_view<Vector, RScaling >& expr) : base(expr) {} 
	};

	template <typename Vector, typename Divisor>
	struct device_expr<divide_by_view<Vector, Divisor> >
	  : device_expr< map_view<tfunctor::divide_by<typename Vector::value_type, Divisor>, Vector> >
	{
	    typedef device_expr< map_view<tfunctor::divide_by<typename Vector::value_type, Divisor>, Vector> > base;
	    device_expr(const divide_by_view<Vector, Divisor >& expr) : base(expr) {} 
	};

	template <typename Vector>
	struct device_expr<conj_view<Vector> >
	  : device_expr<map_view<mtl::sfunctor::conj<typename Vector::value_type>, Vector> >
	{
	    typedef device_expr<map_view<mtl::sfunctor::conj<typename Vector::value_type>, Vector> > base;
	    device_expr(const map_view<mtl::sfunctor::conj<typename Vector::value_type>, Vector>& expr) : base(expr) {}
	};

	template <typename Vector>
	struct device_expr<negate_view<Vector> >
	  : device_expr<map_view<mtl::sfunctor::negate<typename Vector::value_type>, Vector> >
	{
	    typedef device_expr<map_view<mtl::sfunctor::negate<typename Vector::value_type>, Vector> > base;
	    device_expr(const map_view<mtl::sfunctor::negate<typename Vector::value_type>, Vector>& expr) : base(expr) {}
	};


}} // namespace mtl::vector

#endif

#endif // MTL_VECTOR_DEVICE_EXPR_INCLUDE
