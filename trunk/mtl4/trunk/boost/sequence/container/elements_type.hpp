// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef CONTAINER_ELEMENTS_TYPE_DWA200541_HPP
# define CONTAINER_ELEMENTS_TYPE_DWA200541_HPP

# include <boost/sequence/detail/is_container.hpp>
# include <boost/utility/enable_if.hpp>

namespace boost {
namespace sequence { 

namespace elements_type_
{
  // In order to prevent the usual 
  // base template with room for SFINAE on partial specializations
  template <class Sequence>
  struct base<
      Sequence
    , typename boost::enable_if<detail::is_container<Sequence> >::type
  >
  {
      
  };
}

}

#endif // CONTAINER_ELEMENTS_TYPE_DWA200541_HPP
