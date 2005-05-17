// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef MAKE_DWA200541_HPP
# define MAKE_DWA200541_HPP

namespace boost { namespace sequence { namespace detail { 

// Used to manufacture a T for the purposes of compile-time evaluation
// only.  Never defined or invoked.
template <class T> T make();

}}} // namespace boost::sequence::detail

#endif // MAKE_DWA200541_HPP
