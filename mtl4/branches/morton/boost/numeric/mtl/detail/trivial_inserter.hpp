// $COPYRIGHT$

#ifndef MTL_TRIVIAL_INSERTER_INCLUDE
#define MTL_TRIVIAL_INSERTER_INCLUDE

#include <boost/numeric/mtl/operation/update.hpp>


namespace mtl { namespace detail {


// Matrix must have direct write access, i.e. matrix(row, col) must return a non-const reference
template <typename Matrix, typename Updater = mtl::operations::update_store<typename Matrix::value_type> >
struct trivial_inserter
{
    typedef trivial_inserter                            self;
    typedef Matrix                                      matrix_type;
    typedef typename matrix_type::size_type             size_type;
    typedef typename matrix_type::value_type            value_type;
    typedef operations::update_proxy<self, size_type>   proxy_type;
    
    explicit trivial_inserter(matrix_type& matrix) : matrix(matrix) {}

    proxy_type operator() (size_type row, size_type col)
    {
	return proxy_type(*this, row, col);
    }

    void update(size_type row, size_type col, value_type val)
    {
	Updater() (matrix(row, col), val);
    }

  protected:
    matrix_type&         matrix;
};

}} // namespace mtl::detail

#endif // MTL_TRIVIAL_INSERTER_INCLUDE
