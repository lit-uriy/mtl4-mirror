// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_DETAIL_FUNCTION1_DWA200655_HPP
# define BOOST_DETAIL_FUNCTION1_DWA200655_HPP

# include <boost/detail/msvc_array_wknd.hpp>
# include <boost/concept_check.hpp>
# include <boost/type_traits/remove_reference.hpp>

namespace boost { namespace detail { 

// A utility for creating unary function objects that play nicely with
// boost::result_of and that handle the forwarding problem.
// 
// F<A0> is expected to be a stateless function object that accepts an
// argument of type A0&.  It is also expected to have a nested
// ::result_type identical to its return type.
template <template <class A0> class F>
struct function1
{
    template <class A0>
    struct result
    {
        // How adding const to arguments handles rvalues.
        //
        // if A0 is     arg0 is       represents actual argument
        // --------     -------       --------------------------
        // T const&     T const       const T lvalue
        // T&           T             non-const T lvalue
        // T const      T const       const T rvalue  
        // T            T const       non-const T rvalue
        typedef typename remove_reference<A0 const>::type arg0;
        typedef typename F<arg0> impl;
        typedef typename impl::result_type type;
        
        BOOST_CONCEPT_ASSERT((UnaryFunction<impl,type,A0>));
    };
    
    // Handles mutable lvalues
    template <class A0>
    typename result<A0&>::type
    operator()(A0& a0) const
    {
        typedef typename result<A0&>::impl impl;
        return impl()(a0);
    }

    // Handles const lvalues and all rvalues
    template <class A0>
    typename BOOST_MSVC_ARRAY_WKND(
        (A0)
      , ( result<A0 const&> ))
    operator()(A0 const& a0) const
    {
        typedef typename result<A0&>::impl impl;
        return impl()(a0);
    }
};

}} // namespace boost::detail

#endif // BOOST_DETAIL_FUNCTION1_DWA200655_HPP
