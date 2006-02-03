// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef UNSPECIALIZED_DWA200552_HPP
# define UNSPECIALIZED_DWA200552_HPP

# include <boost/type_traits/is_convertible.hpp>
# include <boost/sequence/core/detail/unspecialized.hpp>

namespace boost { namespace sequence { namespace detail {

// This is just a base class to use for templates so we can tell
// whether they've been specialized.
struct unspecialized {};

template <class T>
struct is_unspecialized
  : is_convertible<T*,unspecialized const volatile*>
{
};

}}} // namespace boost::sequence::detail

#endif // UNSPECIALIZED_DWA200552_HPP
