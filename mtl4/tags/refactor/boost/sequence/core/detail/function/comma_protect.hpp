// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_DETAIL_COMMA_PROTECT_DWA20051126_HPP
# define BOOST_SEQUENCE_DETAIL_COMMA_PROTECT_DWA20051126_HPP

namespace boost { namespace sequence { namespace detail { 

// Use
//
//   comma_protect<int(some_metafunction<A1, A2,..., AN>)>
//
// in lieu of
//
//   some_metafunction<A1, A2,..., AN>
//
// when you need to pass some_metafunction<A1, A2,..., AN> as a macro
// argument.
//
template <class F>
struct comma_protect;

template <class R, class F>
struct comma_protect<R(F)>
  : F
{};

}}} // namespace boost::sequence::detail

#endif // BOOST_SEQUENCE_DETAIL_COMMA_PROTECT_DWA20051126_HPP
