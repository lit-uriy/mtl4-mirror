// $COPYRIGHT$

#ifndef MTL_FOR_EACH_NONZERO_INCLUDE
#define MTL_FOR_EACH_NONZERO_INCLUDE

#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>


namespace mtl {


template <typename Vector, typename Functor>
inline void for_each_nonzero(Vector& v, const Functor& f, tag::vector)
{
    typedef typename traits::range_generator<tag::iter::nz, Vector>::type iterator;
    for (iterator i= begin<tag::iter::nz>(v), iend= end<tag::iter::nz>(v); i != iend; ++i)
	f(*i);
}


template <typename Matrix, typename Functor>
inline void for_each_nonzero(Matrix& m, const Functor& f, tag::matrix)
{
    typedef typename traits::range_generator<tag::major, Matrix>::type  cursor_type;
    for (cursor_type cursor = begin<tag::major>(m), cend = end<tag::major>(m); 
	 cursor != cend; ++cursor) 
    {
	typedef typename traits::range_generator<tag::iter::nz, cursor_type>::type iterator;
	for (iterator i= begin<tag::iter::nz>(cursor), iend= end<tag::iter::nz>(cursor); i != iend; ++i)
	    f(*i);
    }
}


template <typename Collection, typename Functor>
inline void for_each_nonzero(Collection& m, const Functor& f)
{
    for_each_nonzero(c, f, typename traits::category<Collection>::type());
}

} // namespace mtl

#endif // MTL_FOR_EACH_NONZERO_INCLUDE
