// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef ELEMENTS_DWA200541_HPP
# define ELEMENTS_DWA200541_HPP

# include <boost/sequence/intrinsics.hpp>
# include <boost/sequence/detail/instance.hpp>

namespace boost {
namespace sequence {

intrinsic::function<intrinsic::elements> const& elements = detail::instance<intrinsic::function<intrinsic::elements> >::object;

}} // namespace boost::sequence

#endif // ELEMENTS_DWA200541_HPP
