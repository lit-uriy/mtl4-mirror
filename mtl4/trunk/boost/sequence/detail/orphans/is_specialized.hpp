// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef IS_SPECIALIZED_DWA200552_HPP
# define IS_SPECIALIZED_DWA200552_HPP

# include <boost/mpl/not.hpp>
# include <boost/sequence/detail/unspecialized.hpp>

namespace boost { namespace sequence { namespace detail { 

template <class T>
struct is_specialized
  : mpl::not_<is_unspecialized<T> >
{};

}}} // namespace boost::sequence::detail

#endif // IS_SPECIALIZED_DWA200552_HPP
