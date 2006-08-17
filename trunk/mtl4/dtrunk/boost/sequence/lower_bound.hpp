// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_LOWER_BOUND_DWA200665_HPP
# define BOOST_SEQUENCE_LOWER_BOUND_DWA200665_HPP

# include <boost/detail/function3.hpp>

# include <boost/spirit/phoenix/core/argument.hpp>
# include <boost/spirit/phoenix/bind.hpp>
# include <boost/spirit/phoenix/function.hpp>
# include <algorithm>

namespace boost { namespace sequence { 

namespace impl
{
  template <class F, class Elements>
  struct comparator
    : boost::compressed_pair<F,Elements>
  {
      comparator(F f, Elements elements)
        : boost::compressed_pair<F,Elements>(f,elements)
      {}

      typedef bool result_type;

      template <class K, class T>
      bool operator()(K const& k, T const& x) const
      {
          return this->first()( this->second()(k), x );
      }
  };

  template <class F, class Elements>
  comparator<F,Elements>
  make_comparator(F f, Elements elements)
  {
      return comparator<F,Elements>(f,elements);
  }
  
  template <class S, class Target, class Cmp>
  struct lower_bound
  {
      typedef typename
        concepts::Sequence<S>::cursor
      result_type;
      
      result_type operator()(S& s, Target& t, Cmp& c) const
      {
//          namespace p = phoenix;
          
          return std::lower_bound(
              sequence::begin(s), sequence::end(s), t
            , impl::make_comparator(c, sequence::elements(s))
# if 0
              p::bind(
                  c
                , p::bind(sequence::elements(s), p::arg1)
                , p::arg2
              )
# endif 
          );
      }
  };
  
}

namespace op
{
  using mpl::_;
  
  struct lower_bound
    : boost::detail::function3<impl::lower_bound<_,_,_> >
  {};
}

namespace
{
  op::lower_bound const& lower_bound = boost::detail::pod_singleton<op::lower_bound>::instance;
}
    
}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_LOWER_BOUND_DWA200665_HPP
