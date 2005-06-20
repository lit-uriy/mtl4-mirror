// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BEGIN_DWA200541_HPP
# define BEGIN_DWA200541_HPP

# include <boost/sequence/intrinsics.hpp>
# include <boost/sequence/detail/instance.hpp>

namespace boost {
namespace sequence {

intrinsic::function<intrinsic::begin> const& begin = detail::instance<intrinsic::function<intrinsic::begin> >::object;
// intrinsic::begin const& begin = detail::instance<intrinsic::begin>::object;

}} // namespace boost::sequence

#endif // BEGIN_DWA200541_HPP
