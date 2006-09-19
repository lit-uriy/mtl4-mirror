// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_INSERT_DWA200665_HPP
# define BOOST_SEQUENCE_INSERT_DWA200665_HPP

# include <boost/sequence/concepts.hpp>
# include <boost/utility/enable_if.hpp>
# include <boost/sequence/tag.hpp>
# include <boost/mpl/placeholders.hpp>
# include <boost/detail/function3.hpp>

namespace boost { namespace sequence { 

namespace concepts
{
  template <class S> struct Sequence;
}

namespace impl
{
  template <class S, class C, class V, class = typename tag<S>::type>
  struct insert
  {
      typedef typename concepts::Sequence<S>::cursor result_type;

      // Assume we're using a counting_iterator adapter and a
      // dereference map, since that's how standard containers are
      // presented as sequences.
      result_type operator()(S& s, C& pos, V& x)
      {
          return result_type( s.insert( pos.base(), x ) );
      }
  };
}

namespace op
{
  using mpl::_;
  struct insert : boost::detail::function3< impl::insert<_,_,_,impl::tag<_> > > {};
}

namespace
{
  op::insert const& insert = boost::detail::pod_singleton<op::insert>::instance;
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_INSERT_DWA200665_HPP
