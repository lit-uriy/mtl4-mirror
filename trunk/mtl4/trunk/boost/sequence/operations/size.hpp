// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_OPERATIONS_SIZE_JDG20060207_HPP
# define BOOST_SEQUENCE_OPERATIONS_SIZE_JDG20060207_HPP

# include <boost/sequence/operations/iterator_range_operations.hpp>
# include <boost/mpl/size_t.hpp>

namespace boost { namespace sequence { namespace intrinsic {

// The default implementation of each intrinsic function object type
// is inherited from the corresponding member of
// operations<Sequence>.  You can of course specialize begin<S>,
// end<S>, and elements<S>, individually, but specializing
// operations<> usually more convenient.

template <class Sequence>
struct size : operations<Sequence>::size {};

template <class Sequence>
struct size<Sequence const>
  : size<Sequence>
{};

template <class Sequence>
struct size<Sequence&>
  : size<Sequence>
{};

template <class T, std::size_t N>
struct size<T[N]>
{
    typedef mpl::size_t<N> type;
    type operator()(T const (&)[N]) { return type(); }
};

template <class T, std::size_t N>
struct size<T const[N]> : size<T[N]> {};

template <class T, std::size_t N>
struct size<boost::array<T,N> >
{
    typedef mpl::size_t<N> type;
    type operator()(boost::array<T,N> const&) { return type(); }
};

template <class T, std::size_t N>
struct size<array<T,N> const> : size<array<T,N> > {};
    
}}} // namespace boost::sequence::intrinsic

#endif // BOOST_SEQUENCE_OPERATIONS_SIZE_JDG20060207_HPP
