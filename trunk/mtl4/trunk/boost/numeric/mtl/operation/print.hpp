// $COPYRIGHT$

#ifndef MTL_PRINT_INCLUDE
#define MTL_PRINT_INCLUDE

#include <iostream>
#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/operation/print_matrix.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>

namespace mtl {

    namespace detail {

	template <typename Value>
	inline std::ostream&
	print(Value const& value, tag::matrix, std::ostream& out= std::cout, int width= 3, int precision= 2)
	{
	    return print_matrix(value, out, width, precision);
	}
    } // namespace detail


    template <typename Value>
    inline std::ostream&
    print(Value const& value, std::ostream& out= std::cout, int width= 3, int precision= 2)
    {
	return detail::print(value, typename traits::category<Value>::type(), out, width, precision);
    }


    template <typename Value, typename Parameter>
    inline std::ostream& operator<< (std::ostream& out, dense2D<Value, Parameter> const& value) 
    {
	return print(value, out);
    }

    template <typename Value, typename Parameter>
    inline std::ostream& operator<< (std::ostream& out, compressed2D<Value, Parameter> const& value) 
    {
	return print(value, out);
    }

    template <typename Value, unsigned long Mask, typename Parameter>
    inline std::ostream& operator<< (std::ostream& out, morton_dense<Value, Mask, Parameter> const& value) 
    {
	return print(value, out);
    }

    template <typename Matrix>
    inline std::ostream& operator<< (std::ostream& out, transposed_view<Matrix> const& value) 
    {
	return print(value, out);
    }

    template <typename Functor, typename Matrix>
    inline std::ostream& operator<< (std::ostream& out, matrix::map_view<Functor, Matrix> const& value) 
    {
	return print(value, out);
    }


// ======================
// use formatting with <<
// ======================


    namespace detail {

	template <typename Matrix>
	struct with_format_t
	{
	    explicit with_format_t(const Matrix& matrix, int width, int precision) 
		: matrix(matrix), width(width), precision(precision)
	    {}

	    const Matrix& matrix;
	    int width, precision;
	};

    } // detail


    template <typename Matrix>
    inline detail::with_format_t<Matrix> with_format(const Matrix& matrix, int width= 3, int precision= 2)
    {
	return detail::with_format_t<Matrix>(matrix, width, precision);
    }


    template <typename Matrix>
    inline std::ostream& operator<< (std::ostream& out, detail::with_format_t<Matrix> const& value) 
    {
	return print(value.matrix, out, value.width, value.precision);
    }
    

} // namespace mtl

#endif // MTL_PRINT_INCLUDE
