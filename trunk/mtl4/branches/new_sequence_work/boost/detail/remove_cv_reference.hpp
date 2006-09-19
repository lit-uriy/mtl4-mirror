// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_DETAIL_REMOVE_CV_REFERENCE_DWA2006919_HPP
# define BOOST_DETAIL_REMOVE_CV_REFERENCE_DWA2006919_HPP

# include <boost/type_traits/remove_reference.hpp>
# include <boost/type_traits/remove_cv.hpp>

namespace boost { namespace detail { 

template <class T>
struct remove_cv_reference
  : remove_cv<typename remove_reference<T>::type>::type
{};

}} // namespace boost::detail

#endif // BOOST_DETAIL_REMOVE_CV_REFERENCE_DWA2006919_HPP
