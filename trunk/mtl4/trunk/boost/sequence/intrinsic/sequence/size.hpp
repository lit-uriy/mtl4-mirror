// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_SIZE_DWA2005815_HPP
# define BOOST_SEQUENCE_SIZE_DWA2005815_HPP

# include <boost/sequence/operations/operations.hpp>
# include <boost/sequence/intrinsic/detail/function/function1.hpp>
# include <boost/sequence/intrinsic/detail/instance.hpp>

namespace boost {
namespace sequence {

BOOST_SEQUENCE_DECLARE_INSTANCE(detail::const_function1<intrinsic::size>, size)

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_SIZE_DWA2005815_HPP
