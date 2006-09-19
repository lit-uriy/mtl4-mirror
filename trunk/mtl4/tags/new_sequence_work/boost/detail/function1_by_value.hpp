// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_DETAIL_FUNCTION1_BY_VALUE_DWA200655_HPP
# define BOOST_DETAIL_FUNCTION1_BY_VALUE_DWA200655_HPP

# include <boost/concept_check.hpp>
# include <boost/detail/remove_cv_reference.hpp>
# include <boost/mpl/apply.hpp>

namespace boost { namespace detail { 

// A utility for creating unary function objects that play nicely with
// boost::result_of and take parameters by value
// 
// mpl::apply<F,A0>::type is expected to be a stateless function
// object that accepts an argument of type A0.  It is also expected
// to have a nested ::result_type identical to its return type.
template <class F>
struct function1_by_value
{
    template <class Signature>
    struct result;

    template <class This, class A0>
    struct result<This(A0)>
    {
        // Strip const-ness and lvalueness.  The formal parameter is a
        // copy of the actual argument, so all constness disappears
        typedef typename remove_cv_reference<A0>::type arg0
        
        typedef typename mpl::apply1<F,arg0>::type impl;
        typedef typename impl::result_type type;
        
        BOOST_CONCEPT_ASSERT((UnaryFunction<impl,type,A0>));
    };
    
    template <class A0>
    typename result<function1_by_value(A0)>::type
    operator()(A0 a0) const
    {
        typedef typename result<function1_by_value(A0)>::impl impl;
        return impl()(a0);
    }
};

}} // namespace boost::detail

#endif // BOOST_DETAIL_FUNCTION1_BY_VALUE_DWA200655_HPP
