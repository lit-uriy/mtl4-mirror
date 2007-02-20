// $COPYRIGHT$

#ifndef MTL_MATRIX_INSERTER_INCLUDE
#define MTL_MATRIX_INSERTER_INCLUDE

#include <boost/numeric/mtl/operations/update.hpp>
#include <boost/numeric/mtl/detail/trivial_inserter.hpp>



namespace mtl {

template <typename Matrix, typename Updater = mtl::operations::update_store<typename Matrix::value_type> >
struct matrix_inserter 
  : public detail::trivial_inserter<Matrix, Updater>
{
    typedef detail::trivial_inserter<Matrix, Updater>     base;

    explicit matrix_inserter(Matrix& matrix) : base(matrix) 
    {
	std::cout << "in default inserter\n";
    }
};

#if 0
template <typename Elt, typename Parameters> class compressed2D;
template <typename Elt, typename Parameters, typename Updater> class compressed2D_inserter;

template <typename Elt, typename Parameters>
struct matrix_inserter<compressed2D<Elt, Parameters>, mtl::operations::update_store<Elt> >
  : compressed2D_inserter<Elt, Parameters, mtl::operations::update_store<Elt> >
{};

template <typename Elt, typename Parameters, typename Updater>
struct matrix_inserter<compressed2D<Elt, Parameters>, Updater>
  : compressed2D_inserter<Elt, Parameters, Updater>
{};
#endif

} // namespace mtl

#endif // MTL_MATRIX_INSERTER_INCLUDE
