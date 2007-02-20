// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_ELEMENTS_DWA200656_HPP
# define BOOST_SEQUENCE_ELEMENTS_DWA200656_HPP

# include <boost/detail/function1.hpp>
# include <boost/property_map/dereference.hpp>
# include <boost/detail/pod_singleton.hpp>

namespace boost { namespace sequence { 

namespace impl
{
  template <class S>
  struct elements
  {
      typedef property_map::dereference result_type;
      
      result_type operator()(S& s) const
      {
          return result_type();
      }
  };
}

namespace op
{
  struct elements : boost::detail::function1<impl::elements> {};
}

namespace
{
  op::elements const& elements = boost::detail::pod_singleton<op::elements>::instance;
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_ELEMENTS_DWA200656_HPP
