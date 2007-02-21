// $COPYRIGHT$

#ifndef MTL_MATRIX_INSERTER_INCLUDE
#define MTL_MATRIX_INSERTER_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/operation/update.hpp>
#include <boost/numeric/mtl/detail/trivial_inserter.hpp>



namespace mtl {

template <typename Matrix, typename Updater = mtl::operations::update_store<typename Matrix::value_type> >
struct matrix_inserter 
  : public detail::trivial_inserter<Matrix, Updater>
{
    typedef detail::trivial_inserter<Matrix, Updater>     base;

    explicit matrix_inserter(Matrix& matrix) : base(matrix) 
    {
      // std::cout << "in default inserter\n";
    }
};


} // namespace mtl

#endif // MTL_MATRIX_INSERTER_INCLUDE
