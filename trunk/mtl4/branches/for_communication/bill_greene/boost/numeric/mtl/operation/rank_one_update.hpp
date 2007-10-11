// $COPYRIGHT$

#ifndef MTL_RANK_ONE_UPDATE_INCLUDE
#define MTL_RANK_ONE_UPDATE_INCLUDE

#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>
#include <boost/numeric/mtl/matrix/inserter.hpp>
#include <boost/numeric/mtl/operation/update.hpp>

namespace mtl {


/// Rank-one update: rank_one_update(A, x, y) computes A+= x * conj(y)^T
/** The current implementation works for column and row vectors (although
    the notation above refers to column vectors). **/
template <typename Matrix, typename VectorX, typename VectorY>
inline void rank_one_update(Matrix& matrix, const VectorX& x, const VectorY& y)
{
    MTL_THROW_IF(num_rows(matrix) != size(x) || num_cols(matrix) != size(y), incompatible_size());

    typedef typename traits::range_generator<tag::nz, VectorX>::type x_cursor;
    typename traits::index<VectorX>::type             index_x(x); 
    typename traits::const_value<VectorX>::type       value_x(x); 

    typedef typename traits::range_generator<tag::nz, VectorY>::type y_cursor;
    typename traits::index<VectorY>::type             index_y(y); 
    typename traits::const_value<VectorY>::type       value_y(y); 

    matrix::inserter<Matrix, operations::update_plus<typename Collection<Matrix>::value_type> > ins(matrix);

    for (x_cursor xc= begin<tag::nz>(x), xend= end<tag::nz>(x); xc != xend; ++xc)
	for (y_cursor yc= begin<tag::nz>(y), yend= end<tag::nz>(y); yc != yend; ++yc)
	    ins(index_x(*xc), index_y(*yc)) << value_x(*xc) * conj(value_y(*yc));
}


} // namespace mtl

#endif // MTL_RANK_ONE_UPDATE_INCLUDE
