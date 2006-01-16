// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef ELEMENTS_DWA200541_HPP
# define ELEMENTS_DWA200541_HPP

# include <boost/sequence/container/elements_map.hpp>
# include <boost/sequence/detail/is_container.hpp>
# include <boost/utility/enable_if.hpp>

namespace boost {
namespace sequence {

template <class C>
typename enable_if<
    detail::is_container<C>
  , container::elements_map<C>
>::type
inline
elements(C&)
{
    return container::elements_map<C>();
}

}} // namespace boost::sequence

#endif // ELEMENTS_DWA200541_HPP
