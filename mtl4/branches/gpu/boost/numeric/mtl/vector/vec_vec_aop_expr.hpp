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


#ifndef MTL_VEC_VEC_AOP_EXPR_INCLUDE
#define MTL_VEC_VEC_AOP_EXPR_INCLUDE

#include <boost/mpl/bool.hpp>
#include <boost/numeric/mtl/config.hpp>
#include <boost/numeric/mtl/vector/vec_expr.hpp>
#include <boost/numeric/mtl/vector/device_expr.hpp>
#include <boost/numeric/mtl/operation/static_size.hpp>
#include <boost/numeric/mtl/operation/sfunctor.hpp>
#include <boost/numeric/mtl/operation/check.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/is_static.hpp>
#include <boost/numeric/mtl/cuda/launch_function.hpp>

#define BL_SIZE 256

namespace mtl { namespace vector {

    namespace impl {

	template <unsigned long Index, unsigned long Max, typename SFunctor>
	struct assign
	{
	    typedef assign<Index+1, Max, SFunctor>     next;

	    template <class E1, class E2>
	    static inline void apply(E1& first, const E2& second)
	    {
		SFunctor::apply( first(Index), second(Index) );
		next::apply( first, second );
	    }
	};

	template <unsigned long Max, typename SFunctor>
	struct assign<Max, Max, SFunctor>
	{
	    template <class E1, class E2>
	    static inline void apply(E1& first, const E2& second)
	    {
		SFunctor::apply( first(Max), second(Max) );
	    }
	};
    }

// Generic assign operation expression template for vectors
// Model of VectorExpression
template <class E1, class E2, typename SFunctor>
struct vec_vec_aop_expr 
  :  public vec_expr< vec_vec_aop_expr<E1, E2, SFunctor> >
{
    typedef vec_expr< vec_vec_aop_expr<E1, E2, SFunctor> >  expr_base;
    typedef vec_vec_aop_expr<E1, E2, SFunctor>   self;
    typedef typename E1::value_type              value_type;
    
    // temporary solution
    typedef typename E1::size_type               size_type;
    //typedef typename utilities::smallest< typename E1::size_type, typename E2::size_type >::type                          size_type ;

    typedef value_type reference_type ;

    typedef E1 first_argument_type ;
    typedef E2 second_argument_type ;
    
    vec_vec_aop_expr( first_argument_type& v1, second_argument_type const& v2 )
      : first( v1 ), second( v2 ), delayed_assign( false )
    {
	second.delay_assign();
    }

#if 0
    void assign(boost::mpl::false_)
    {
	// If target is constructed by default it takes size of source
	if (first.size() == 0) first.change_dim(second.size());

	// If sizes are different for any other reason, it's an error
	// std::cerr << "~vec_vec_aop_expr() " << first.size() << "  " << second.size() << "\n";
	MTL_DEBUG_THROW_IF(first.size() != second.size(), incompatible_size());

	for (size_type i= 0; i < first.size(); ++i)
	    SFunctor::apply( first(i), second(i) );
    }

    void assign(boost::mpl::true_)
    {
	// We cannot resize, only check
	MTL_DEBUG_THROW_IF(first.size() != second.size(), incompatible_size());
	impl::assign<1, static_size<E1>::value, SFunctor>::apply(first, second);
    }
#endif

    void to_device() const { first.to_device(); second.to_device(); }
    void to_host() const { first.to_host(); second.to_host(); }
    bool valid_device() const { return first.valid_device() && second.valid_device(); }
    bool valid_host() const { return first.valid_host() && second.valid_host(); }

    int gridDimx(int dim) { return std::min(65535, (dim+BL_SIZE-1)/BL_SIZE); }
   
   void compute_on_host()
   {
       for (size_type i= 0; i < size(first); ++i)
	   SFunctor::apply( const_cast<first_argument_type&>(first)(i), second(i) );
       // Slower, at least on gcc
       // assign(traits::is_static<E1>());       
   }
   
#ifdef MTL_HAS_CUDA
    struct kernel
    {
	kernel(E1& first, const E2& second) : first(first), second(second), n(size(first)) { }
	
	__device__ void operator()()
	{
	    const size_type grid_size = blockDim.x * gridDim.x, 
		            id= blockIdx.x * blockDim.x + threadIdx.x,
			    blocks= n / grid_size,  nn= blocks * grid_size;
	    value_type* p= const_cast<value_type*>(first.dptr);
	    
	    for (size_type i= id; i < nn; i+= grid_size) 
		SFunctor::apply(p[i], second[i]);
	    if (nn + id < n) 
		SFunctor::apply(p[nn + id], second[nn + id]);
	}
      private:
	device_expr<E1>   first;
	device_expr<E2>   second;
	size_type         n;
    };

#endif // MTL_HAS_CUDA
   
   ~vec_vec_aop_expr()
    {
	if (!delayed_assign) {
	    // If target is constructed by default it takes size of source
	    //if (size(first) == 0) first.change_dim(size(second));

	    // If sizes are different for any other reason, it's an error
	    MTL_DEBUG_THROW_IF(size(first) != size(second), incompatible_size());

#ifdef MTL_HAS_CUDA
	    if (meet_data(first, second))
		compute_on_host();
	    else {
		dim3 dimGrid(gridDimx(size(first))), dimBlock(BL_SIZE); // temporary sol.
		kernel k(const_cast<first_argument_type&>(first), second);
		cuda::launch_function<<<dimGrid, dimBlock>>>(k);
		//cuda::launch_function<<<dim3(1), dim3(10)>>>(k);
	    }
#else
	    compute_on_host();
#endif
	}
    }
    
    void delay_assign() const { delayed_assign= true; }

    friend size_type inline size(const self& x)
    {
	assert( size(x.first) == 0 || size(x.first) == size(x.second) );
	return size(x.first);
    }


    MTL_PU value_type& operator() ( size_type i ) const {
	check( delayed_assign );
	return SFunctor::apply( first(i), second(i) );
    }

    MTL_PU value_type& operator[] ( size_type i ) const{
	check( delayed_assign );
	return SFunctor::apply( first(i), second(i) );
    }

  private:
     first_argument_type const&          first ;
     second_argument_type const&         second ;
     mutable bool                        delayed_assign;
  } ; // vec_vec_aop_expr

} } // Namespace mtl::vector





#endif


