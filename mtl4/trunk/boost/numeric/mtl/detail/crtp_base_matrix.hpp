// $COPYRIGHT$

#ifndef MTL_CRTP_BASE_MATRIX_INCLUDE
#define MTL_CRTP_BASE_MATRIX_INCLUDE

#include <boost/numeric/mtl/operation/matrix_bracket.hpp>

namespace mtl { namespace detail {

template <typename Matrix, typename ValueType, typename SizeType>
struct const_crtp_base_matrix
{    
    operations::bracket_proxy<Matrix, const Matrix&, ValueType>
    operator[] (SizeType row) const
    {
	return operations::bracket_proxy<Matrix, const Matrix&, ValueType>(static_cast<const Matrix&>(*this), row);
    }
};

template <typename Matrix, typename ValueType, typename SizeType>
struct crtp_base_matrix // : public const_crtp_base_matrix<Matrix, ValueType, SizeType>
{    
    operations::bracket_proxy<Matrix, const Matrix&, const ValueType&>
    operator[] (SizeType row) const
    {
        return operations::bracket_proxy<Matrix, const Matrix&, const ValueType&>(static_cast<const Matrix&>(*this), row);
    }

    operations::bracket_proxy<Matrix, Matrix&, ValueType&>
    operator[] (SizeType row)
    {
        return operations::bracket_proxy<Matrix, Matrix&, ValueType&>(static_cast<Matrix&>(*this), row);
    }
};


}} // namespace mtl::detail

#endif // MTL_CRTP_BASE_MATRIX_INCLUDE
