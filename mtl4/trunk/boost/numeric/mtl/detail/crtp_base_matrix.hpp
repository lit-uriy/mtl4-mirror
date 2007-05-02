// $COPYRIGHT$

#ifndef MTL_CRTP_BASE_MATRIX_INCLUDE
#define MTL_CRTP_BASE_MATRIX_INCLUDE

#include <iostream>
#include <boost/numeric/mtl/operation/print.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/operation/matrix_bracket.hpp>
#include <boost/numeric/mtl/operation/copy.hpp>
#include <boost/numeric/mtl/matrix/all_mat_expr.hpp>

namespace mtl { namespace detail {

template <typename Matrix, typename ValueType, typename SizeType>
struct crtp_matrix_assign
{
    /// Assign matrix expressions by copying except for some special expressions
    template <typename MatrixSrc>
    Matrix& operator=(const matrix::mat_expr<MatrixSrc>& src)
    {
	matrix_copy(src.ref, static_cast<Matrix&>(*this));
	return static_cast<Matrix&>(*this);
    }

    /// Assign sum by assigning first argument and adding second
    /** Note that this is more special then assigning arbitrary expressions including matrices itself
	because matrix::mat_mat_plus_expr <E1, E2> is a derived class from matrix::mat_expr < MatrixSrc >. **/
    template <typename E1, typename E2>
    Matrix& operator=(const matrix::mat_mat_plus_expr<E1, E2>& src)
    {
	static_cast<Matrix&>(*this)= src.first;
	static_cast<Matrix&>(*this)+= src.second;

	return static_cast<Matrix&>(*this);
    }

    /// Assign difference by assigning first argument and subtracting second
    /** Note that this is more special then assigning arbitrary expressions including matrices itself
	because matrix::mat_mat_minus_expr <E1, E2> is a derived class from matrix::mat_expr < MatrixSrc >. **/
    template <typename E1, typename E2>
    Matrix& operator=(const matrix::mat_mat_minus_expr<E1, E2>& src)
    {
	static_cast<Matrix&>(*this)= src.first;
	static_cast<Matrix&>(*this)-= src.second;

	return static_cast<Matrix&>(*this);
    }

    /// Assign-add matrix expressions by incrementally copying except for some special expressions
    template <typename MatrixSrc>
    Matrix& operator+=(const matrix::mat_expr<MatrixSrc>& src)
    {
	matrix_copy_plus(src.ref, static_cast<Matrix&>(*this));
	return static_cast<Matrix&>(*this);
    }

    /// Assign-add sum by adding both arguments
    /** Note that this is more special then assigning arbitrary expressions including matrices itself
	because matrix::mat_mat_plus_expr <E1, E2> is a derived class from 
	matrix::mat_expr < MatrixSrc >. **/
    template <typename E1, typename E2>
    Matrix& operator+=(const matrix::mat_mat_plus_expr<E1, E2>& src)
    {
	static_cast<Matrix&>(*this)+= src.first;
	static_cast<Matrix&>(*this)+= src.second;

	return static_cast<Matrix&>(*this);
    }

    /// Assign-add difference by adding first argument and subtracting the second one
    /** Note that this is more special then assigning arbitrary expressions including matrices itself
	because matrix::mat_mat_minus_expr <E1, E2> is a derived class from 
	matrix::mat_expr < MatrixSrc >. **/
    template <typename E1, typename E2>
    Matrix& operator+=(const matrix::mat_mat_minus_expr<E1, E2>& src)
    {
	static_cast<Matrix&>(*this)+= src.first;
	static_cast<Matrix&>(*this)-= src.second;

	return static_cast<Matrix&>(*this);
    }

    /// Assign-subtract matrix expressions by decrementally copying except for some special expressions
    template <typename MatrixSrc>
    Matrix& operator-=(const matrix::mat_expr<MatrixSrc>& src)
    {
	matrix_copy_minus(src.ref, static_cast<Matrix&>(*this));
	return static_cast<Matrix&>(*this);
    }

    /// Assign-subtract sum by adding both arguments
    /** Note that this is more special then assigning arbitrary expressions including matrices itself
	because matrix::mat_mat_plus_expr <E1, E2> is a derived class from 
	matrix::mat_expr < MatrixSrc >. **/
    template <typename E1, typename E2>
    Matrix& operator-=(const matrix::mat_mat_plus_expr<E1, E2>& src)
    {
	static_cast<Matrix&>(*this)-= src.first;
	static_cast<Matrix&>(*this)-= src.second;

	return static_cast<Matrix&>(*this);
    }

    /// Assign-subtracting difference by subtracting first argument and adding the second one
    /** Note that this is more special then assigning arbitrary expressions including matrices itself
	because matrix::mat_mat_minus_expr <E1, E2> is a derived class from 
	matrix::mat_expr < MatrixSrc >. **/
    template <typename E1, typename E2>
    Matrix& operator-=(const matrix::mat_mat_minus_expr<E1, E2>& src)
    {
	static_cast<Matrix&>(*this)-= src.first;
	static_cast<Matrix&>(*this)+= src.second;

	return static_cast<Matrix&>(*this);
    }

};



template <typename Matrix, typename ValueType, typename SizeType>
struct const_crtp_matrix_bracket
{    
    operations::bracket_proxy<Matrix, const Matrix&, ValueType>
    operator[] (SizeType row) const
    {
	return operations::bracket_proxy<Matrix, const Matrix&, ValueType>(static_cast<const Matrix&>(*this), row);
    }
};

template <typename Matrix, typename ValueType, typename SizeType>
struct crtp_matrix_bracket 
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

template <typename Matrix, typename ValueType, typename SizeType>
struct const_crtp_base_matrix
    : public const_crtp_matrix_bracket<Matrix, ValueType, SizeType>
{};

template <typename Matrix, typename ValueType, typename SizeType>
struct crtp_base_matrix 
    : public crtp_matrix_bracket<Matrix, ValueType, SizeType>,
      public crtp_matrix_assign<Matrix, ValueType, SizeType>
{};



}} // namespace mtl::detail

#endif // MTL_CRTP_BASE_MATRIX_INCLUDE
