// $COPYRIGHT$

#ifndef MTL_TRIVIAL_INSERTER_INCLUDE
#define MTL_TRIVIAL_INSERTER_INCLUDE

#include <boost/numeric/mtl/operation/update.hpp>
#include <boost/numeric/mtl/matrix/element_matrix.hpp> 
#include <boost/numeric/mtl/matrix/element_array.hpp> 

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
    
    explicit trivial_inserter(matrix_type& matrix, size_type) : matrix(matrix) {}

    proxy_type operator() (size_type row, size_type col)
    {
	return proxy_type(*this, row, col);
    }

    template <typename Value>
    void update(size_type row, size_type col, Value val)
    {
	Updater() (matrix(row, col), val);
    }

    template <typename EMatrix, typename Rows, typename Cols>
    self& operator<< (const matrix::element_matrix_t<EMatrix, Rows, Cols>& elements)
    {
	for (unsigned ri= 0; ri < elements.rows.size(); ri++)
	    for (unsigned ci= 0; ci < elements.cols.size(); ci++)
		update (elements.rows[ri], elements.cols[ci], elements.matrix(ri, ci));
	return *this;
    }

    template <typename EMatrix, typename Rows, typename Cols>
    self& operator<< (const matrix::element_array_t<EMatrix, Rows, Cols>& elements)
    {
	for (unsigned ri= 0; ri < elements.rows.size(); ri++)
	    for (unsigned ci= 0; ci < elements.cols.size(); ci++)
		update (elements.rows[ri], elements.cols[ci], elements.array[ri][ci]);
	return *this;
    }

  protected:
    matrix_type&         matrix;
};

}} // namespace mtl::detail

#endif // MTL_TRIVIAL_INSERTER_INCLUDE
