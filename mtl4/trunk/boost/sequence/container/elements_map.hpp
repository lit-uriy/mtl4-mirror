// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef ELEMENTS_MAP_DWA200541_HPP
# define ELEMENTS_MAP_DWA200541_HPP

# include <boost/sequence/detail/transfer_cv.hpp>

namespace boost { namespace sequence {
namespace container
{ 
  template <class Container>
  struct elements_map
  {
      map(Container& c) {}

      typename detail::transfer_cv<Container, typename Container::value_type>::type&
      operator()(typename Container::const_iterator p)
      {
          return *p;
      }


      typename detail::transfer_cv<Container, typename Container::value_type>::type&
      operator()(typename Container::iterator p)
      {
          return *p;
      }
  };
}
}} // namespace boost::sequence

#endif // ELEMENTS_MAP_DWA200541_HPP
