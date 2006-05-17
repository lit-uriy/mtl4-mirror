// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_PROPERTY_MAP_COMPOSE_DWA2006516_HPP
# define BOOST_PROPERTY_MAP_COMPOSE_DWA2006516_HPP

# include <boost/compressed_pair.hpp>
# include <boost/detail/callable.hpp>
# include <boost/utility/result_of.hpp>
# include <boost/detail/function2.hpp>
# include <boost/detail/pod_singleton.hpp>
# include <boost/mpl/placeholders.hpp>
# include <boost/type_traits/remove_const.hpp>
# include <boost/type_traits/is_reference.hpp>
# include <boost/utility/enable_if.hpp>

# include <boost/mpl/assert.hpp>

namespace boost { namespace property_map { 

template <class OuterFn, class InnerFn>
struct compose_
  : boost::detail::callable<compose_<OuterFn,InnerFn> >
{
    template <class Signature, class enabled = void>
    struct result {};
    
    compose_(OuterFn const& outer = OuterFn(), InnerFn const& inner = InnerFn())
      : f(outer,inner)
    {}
    
    template <class F, class A0>
    struct result<F(A0)>
      : result_of<
            OuterFn const( typename result_of<InnerFn const(A0)>::type )
        >
    {};

    template <class A0>
    typename result_of<compose_ const(A0&)>::type
    call(A0& x0) const
    {
        return this->f.first()( this->f.second()( x0 ) );
    }

    template <class F, class A0, class A1>
    struct result<
        F(A0,A1)

# if !BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1400))
        // Disable this specialization if we're not an lvalue property
        // map.  This will yield a disabled binary function call
        // operator in callable.
      , typename enable_if<
            is_reference< typename result<F(A0)>::type >
        >::type
# endif 
    >
    {
        typedef void type;
    };

    template <class A0, class A1>
    void call(A0& x0, A1& x1) const
    {
# if BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1400))
        BOOST_MPL_ASSERT((is_reference< typename result<compose_ const(A0)>::type >));
# endif 
        this->f.first()( this->f.second()( x0 ) ) = x1;
    }
    
 private:
    compressed_pair<OuterFn,InnerFn> f;
};

namespace impl
{
  template <class  OuterFn, class InnerFn>
  struct compose
  {
      typedef compose_<
          typename remove_const<OuterFn>::type
        , typename remove_const<InnerFn>::type
      > result_type;
      
      result_type operator()(OuterFn& o, InnerFn& i) const
      {
          return result_type(o,i);
      }
  };
}

namespace op
{
  using mpl::_;
  struct compose : boost::detail::function2<impl::compose<_,_> > {};
}

namespace
{
  op::compose const& compose = boost::detail::pod_singleton<op::compose>::instance;
}

}} // namespace boost::property_map

#endif // BOOST_PROPERTY_MAP_COMPOSE_DWA2006516_HPP
