// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_NEXT_DWA2005815_HPP
# define BOOST_SEQUENCE_NEXT_DWA2005815_HPP

# include <boost/sequence/intrinsics.hpp>
# include <boost/sequence/detail/function1.hpp>
# include <boost/sequence/detail/instance.hpp>

namespace boost {
namespace sequence {

BOOST_SEQUENCE_DECLARE_INSTANCE(detail::const_function1<intrinsic::next>, next)

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_NEXT_DWA2005815_HPP
