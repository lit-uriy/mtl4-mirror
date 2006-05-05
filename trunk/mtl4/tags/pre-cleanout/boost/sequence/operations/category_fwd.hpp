// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_CATEGORY_FWD_DWA200559_HPP
# define BOOST_SEQUENCE_CATEGORY_FWD_DWA200559_HPP

namespace boost { namespace sequence { 

template <class Sequence>
struct category_impl;

template <class Sequence>
struct category
  : category_impl<Sequence>
{};

// In general, a const T has the same category as T.
template <class T>
struct category<T const>
  : category<T> {};

// In general, a reference to T has the same category as T.
template <class T>
struct category<T&>
  : category<T> {};

struct sequence_tag {};
struct o1_size_tag : sequence_tag {};

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_CATEGORY_FWD_DWA200559_HPP
