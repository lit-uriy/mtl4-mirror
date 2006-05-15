// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// #include guards intentionally disabled.
// #ifndef BOOST_DETAIL_FUNCTION_N_DWA2006514_HPP
// # define BOOST_DETAIL_FUNCTION_N_DWA2006514_HPP

#include <boost/mpl/apply.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/size.hpp>

namespace boost { namespace detail { 

# define BOOST_DETAIL_function_arg(z, n, _)             \
    typedef typename remove_reference<                  \
        typename add_const< BOOST_PP_CAT(A,n) >::type   \
    >::type BOOST_PP_CAT(arg,n);
    
template <class F>
struct BOOST_PP_CAT(function,n)
{
    template <class Signature>
    struct result;

    
    template <class This BOOST_PP_ENUM_TRAILING_PARAMS(n, class A)>
    struct result<This(BOOST_PP_ENUM_PARAMS(n, A))>
    {
        BOOST_PP_REPEAT(n, BOOST_DETAIL_function_arg, ~)
        
        typedef typename mpl::BOOST_PP_CAT(apply,n)<
            F BOOST_PP_ENUM_TRAILING_PARAMS(n,arg)
        >::type impl;
    
        typedef typename impl::result_type type;
        
//      BOOST_CONCEPT_ASSERT((BinaryFunction<impl,type,A0,A1>));
    };

# define arg_type(r,_,i,is_const)                                               \
    BOOST_PP_COMMA_IF(i) BOOST_PP_CAT(A,i) BOOST_PP_CAT(const_if,is_const) &
    
# define result_(r,n,constness)                                 \
    typename result<                                            \
        BOOST_PP_CAT(function,n)(                               \
            BOOST_PP_SEQ_FOR_EACH_I_R(r,arg_type,~,constness)   \
        )                                                       \
    >
        
# define param(r,_,i,is_const) BOOST_PP_COMMA_IF(i)                             \
        BOOST_PP_CAT(A,i) BOOST_PP_CAT(const_if,is_const) & BOOST_PP_CAT(x,i)
    
# define param_list(r,n,constness)                  \
    BOOST_PP_SEQ_FOR_EACH_I_R(r,param,~,constness)
        
# define call_operator(r, constness)                    \
    template <BOOST_PP_ENUM_PARAMS(n, class A)>         \
        result_(r, n, constness)::type                  \
    operator()( param_list(r, n, constness) ) const     \
    {                                                   \
        typedef result_(r, n, constness)::impl impl;    \
        return impl()(BOOST_PP_ENUM_PARAMS(n,x));       \
    }

# define const_if0
# define const_if1 const

# define bits(z, n, _) ((0)(1))

    BOOST_PP_SEQ_FOR_EACH_PRODUCT(
        call_operator
      , BOOST_PP_REPEAT(n, bits, ~)
    )

# undef bits
# undef const_if1
# undef const_if0
# undef call_operator
# undef param_list
# undef param
# undef result_
# undef arg_type

# undef n
};

}} // namespace boost::detail

//#endif // BOOST_DETAIL_FUNCTION_N_DWA2006514_HPP
