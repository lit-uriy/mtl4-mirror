// $COPYRIGHT$

#ifndef MTL_ELEMENT_ARRAY_INCLUDE
#define MTL_ELEMENT_ARRAY_INCLUDE

namespace mtl {

    namespace matrix {

	template <typename Array, typename Rows, typename Cols>
	struct element_array_t
	{
	    explicit element_array_t(const Array& array, const Rows& rows, const Cols& cols)
		: array(array), rows(rows), cols(cols)
	    {}
	    
	    const Array& array;
	    const Rows&  rows;
	    const Cols&  cols;
	};
    }

template <typename Array, typename Rows, typename Cols>
matrix::element_array_t<Array, Rows, Cols>
inline element_array(const Array& array, const Rows& rows, const Cols& cols)
{
    return matrix::element_array_t<Array, Rows, Cols>(array, rows, cols);
}

template <typename Array, typename Rows>
matrix::element_array_t<Array, Rows, Rows>
inline element_array(const Array& array, const Rows& rows)
{
    return matrix::element_array_t<Array, Rows, Rows>(array, rows, rows);
}

} // namespace mtl

#endif // MTL_ELEMENT_ARRAY_INCLUDE
