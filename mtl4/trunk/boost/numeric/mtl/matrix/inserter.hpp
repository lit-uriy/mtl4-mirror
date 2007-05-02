// $COPYRIGHT$

#ifndef MTL_MATRIX_INSERTER_INCLUDE
#define MTL_MATRIX_INSERTER_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/operation/update.hpp>
#include <boost/numeric/mtl/detail/trivial_inserter.hpp>



namespace mtl { namespace matrix {

template <typename Matrix, 
	  typename Updater = mtl::operations::update_store<typename Matrix::value_type> >
struct inserter 
  : public mtl::detail::trivial_inserter<Matrix, Updater>
{
    typedef mtl::detail::trivial_inserter<Matrix, Updater>     base;

    explicit inserter(Matrix& matrix) : base(matrix) 
    {
      // std::cout << "in default inserter\n";
    }
};


template <typename Elt, typename Parameters, typename Updater>
struct inserter<compressed2D<Elt, Parameters>, Updater>
  : compressed2D_inserter<Elt, Parameters, Updater>
{
    typedef compressed2D<Elt, Parameters>     matrix_type;
    typedef typename matrix_type::size_type   size_type;
    typedef compressed2D_inserter<Elt, Parameters, Updater > base;

    explicit inserter(matrix_type& matrix, size_type slot_size = 5) : base(matrix, slot_size) {}
};


}} // namespace mtl::matrix

#endif // MTL_MATRIX_INSERTER_INCLUDE
