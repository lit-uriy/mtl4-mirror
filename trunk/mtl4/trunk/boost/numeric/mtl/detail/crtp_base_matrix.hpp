// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_CRTP_BASE_MATRIX_INCLUDE
#define MTL_CRTP_BASE_MATRIX_INCLUDE

#include <iostream>
#include <boost/utility/enable_if.hpp>
#include <boost/numeric/mtl/operation/print.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/operation/matrix_bracket.hpp>
#include <boost/numeric/mtl/operation/copy.hpp>
#include <boost/numeric/mtl/operation/mult.hpp>
#include <boost/numeric/mtl/operation/right_scale_inplace.hpp>
#include <boost/numeric/mtl/operation/divide_by_inplace.hpp>
#include <boost/numeric/mtl/matrix/all_mat_expr.hpp>
#include <boost/numeric/mtl/matrix/diagonal_setup.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>
#include <boost/numeric/mtl/operation/mult_assign_mode.hpp>
#include <boost/numeric/mtl/operation/compute_factors.hpp>

namespace mtl { namespace detail {

template <typename Source, typename Matrix>
struct crtp_assign 
{
    Matrix& operator()(const Source& source, Matrix& matrix)
    {
		return assign(source, matrix, typename ashape::ashape<Source>::type());
    }
private:
    /// Assign scalar to a matrix by setting the matrix to a multiple of unity matrix
    /** Uses internally \sa diagonal_setup, for details see there. **/
    Matrix& assign(const Source& source, Matrix& matrix, ashape::scal)
    {
	MTL_DEBUG_THROW_IF(num_rows(matrix) * num_cols(matrix) == 0, 
			   range_error("Trying to initialize a 0 by 0 matrix with a value"));
	matrix::diagonal_setup(matrix, source);
	return matrix;
    }

    /// Assign matrix expressions by copying except for some special expressions
    Matrix& assign(const Source& source, Matrix& matrix, typename ashape::ashape<Matrix>::type)
    {
		// Self-assignment between different types shouldn't happen.	
		matrix.checked_change_dim(num_rows(source), num_cols(source));
		matrix_copy(source, matrix);
		return matrix;
    }
};



/// Assign sum by assigning first argument and adding second
/*  Note that this is more special then assigning arbitrary expressions including matrices itself
    because matrix::mat_mat_plus_expr <E1, E2> is a derived class from matrix::mat_expr < MatrixSrc >. **/
template <typename E1, typename E2, typename Matrix>
struct crtp_assign<matrix::mat_mat_plus_expr<E1, E2>, Matrix> 
{
    Matrix& operator()(const matrix::mat_mat_plus_expr<E1, E2>& src, Matrix& matrix)
    {
		matrix.checked_change_dim(num_rows(src.first), num_cols(src.first));
		matrix= src.first;
		matrix+= src.second;
		return matrix;
    }
};

/// Assign difference by assigning first argument and subtracting second
/*  Note that this is more special then assigning arbitrary expressions including matrices itself
    because matrix::mat_mat_minus_expr <E1, E2> is a derived class from matrix::mat_expr < MatrixSrc >. **/
template <typename E1, typename E2, typename Matrix>
struct crtp_assign<matrix::mat_mat_minus_expr<E1, E2>, Matrix> 
{
    Matrix& operator()(const matrix::mat_mat_minus_expr<E1, E2>& src, Matrix& matrix)
    {
		matrix.checked_change_dim(num_rows(src.first), num_cols(src.first));
		matrix= src.first;
		matrix-= src.second;
		return matrix;
    }
};

/// Assign product by calling mult
template <typename E1, typename E2, typename Matrix>
struct crtp_assign<matrix::mat_mat_times_expr<E1, E2>, Matrix> 
{
    Matrix& operator()(const matrix::mat_mat_times_expr<E1, E2>& src, Matrix& matrix)
    {
		operation::compute_factors<Matrix, matrix::mat_mat_times_expr<E1, E2> > factors(src);
		matrix.checked_change_dim(num_rows(factors.first), num_cols(factors.second));
		mult(factors.first, factors.second, matrix);
		return matrix;
    }
}; 

/// Assign c-style 2D-array, because it's easier to initialize.
template <typename Value, unsigned Rows, unsigned Cols, typename Matrix>
struct crtp_assign<Value[Rows][Cols], Matrix>
{
    Matrix& operator()(const Value src[Rows][Cols], Matrix& matrix)
    {
		typedef typename Collection<Matrix>::size_type size_type;

		matrix.checked_change_dim(Rows, Cols);
		matrix::inserter<Matrix>  ins(matrix);

		for (size_type r= 0; r < Rows; ++r)
			for (size_type c= 0; c < Cols; ++c)
				ins(r, c) << src[r][c];
		return matrix;
	}
};




	
/// Assign-add matrix expressions by incrementally copying except for some special expressions
template <typename Source, typename Matrix>
struct crtp_plus_assign 
{
    Matrix& operator()(const Source& source, Matrix& matrix)
    {
		return assign(source, matrix, typename ashape::ashape<Source>::type());
    }
  private:
    Matrix& assign(const Source& source, Matrix& matrix, typename ashape::ashape<Matrix>::type)
    {
		matrix.checked_change_dim(num_rows(source), num_cols(source));
		matrix_copy_plus(source, matrix);
		return matrix;
    }
};

/// Assign-add sum by adding both arguments
/** Note that this is more special then assigning arbitrary expressions including matrices itself
	because matrix::mat_mat_plus_expr <E1, E2> is a derived class from 
	matrix::mat_expr < MatrixSrc >. **/
template <typename E1, typename E2, typename Matrix>
struct crtp_plus_assign<matrix::mat_mat_plus_expr<E1, E2>, Matrix> 
{
    Matrix& operator()(const matrix::mat_mat_plus_expr<E1, E2>& src, Matrix& matrix)
    {
		matrix.checked_change_dim(num_rows(src.first), num_cols(src.first));
		matrix+= src.first;
		matrix+= src.second;
		return matrix;
    }
};

template <typename E1, typename E2, typename Matrix>
struct crtp_plus_assign<matrix::mat_mat_minus_expr<E1, E2>, Matrix> 
{
    Matrix& operator()(const matrix::mat_mat_minus_expr<E1, E2>& src, Matrix& matrix)
    {
		matrix.checked_change_dim(num_rows(src.first), num_cols(src.first));
		matrix+= src.first;
		matrix-= src.second;
		return matrix;
    }
};

template <typename E1, typename E2, typename Matrix>
struct crtp_plus_assign<matrix::mat_mat_times_expr<E1, E2>, Matrix> 
{
    Matrix& operator()(const matrix::mat_mat_times_expr<E1, E2>& src, Matrix& matrix)
    {
		operation::compute_factors<Matrix, matrix::mat_mat_times_expr<E1, E2> > factors(src);
		matrix.checked_change_dim(num_rows(factors.first), num_cols(factors.second));
		gen_mult(factors.first, factors.second, matrix, assign::plus_sum(), tag::matrix(), tag::matrix(), tag::matrix());
		return matrix;
    }
};


/// Assign-subtract matrix expressions by decrementally copying except for some special expressions
template <typename Source, typename Matrix>
struct crtp_minus_assign 
{
    Matrix& operator()(const Source& source, Matrix& matrix)
    {
		return assign(source, matrix, typename ashape::ashape<Source>::type());
    }
private:
    Matrix& assign(const Source& source, Matrix& matrix, typename ashape::ashape<Matrix>::type)
    {
		matrix.checked_change_dim(num_rows(source), num_cols(source));
		matrix_copy_minus(source, matrix);
		return matrix;
    }
};

/// Assign-subtract sum by adding both arguments
/** Note that this is more special then assigning arbitrary expressions including matrices itself
	because matrix::mat_mat_plus_expr <E1, E2> is a derived class from 
	matrix::mat_expr < MatrixSrc >. **/
template <typename E1, typename E2, typename Matrix>
struct crtp_minus_assign<matrix::mat_mat_plus_expr<E1, E2>, Matrix> 
{
    Matrix& operator()(const matrix::mat_mat_plus_expr<E1, E2>& src, Matrix& matrix)
    {
		matrix.checked_change_dim(num_rows(src.first), num_cols(src.first));
		matrix-= src.first;
		matrix-= src.second;
		return matrix;
    }
};

/// Assign-subtracting difference by subtracting first argument and adding the second one
/** Note that this is more special then assigning arbitrary expressions including matrices itself
	because matrix::mat_mat_minus_expr <E1, E2> is a derived class from 
	matrix::mat_expr < MatrixSrc >. **/
template <typename E1, typename E2, typename Matrix>
struct crtp_minus_assign<matrix::mat_mat_minus_expr<E1, E2>, Matrix> 
{
    Matrix& operator()(const matrix::mat_mat_minus_expr<E1, E2>& src, Matrix& matrix)
    {
		matrix.checked_change_dim(num_rows(src.first), num_cols(src.first));
		matrix-= src.first;
		matrix+= src.second;
		return matrix;
    }
};

/// Assign-subtract product by calling gen_mult
/** Note that this does not work for arbitrary expressions. **/
template <typename E1, typename E2, typename Matrix>
struct crtp_minus_assign<matrix::mat_mat_times_expr<E1, E2>, Matrix> 
{
    Matrix& operator()(const matrix::mat_mat_times_expr<E1, E2>& src, Matrix& matrix)
    {
		operation::compute_factors<Matrix, matrix::mat_mat_times_expr<E1, E2> > factors(src);
		matrix.checked_change_dim(num_rows(factors.first), num_cols(factors.second));
		gen_mult(factors.first, factors.second, matrix, assign::minus_sum(), tag::matrix(), tag::matrix(), tag::matrix());
		return matrix;
    }
};



/// Base class to provide matrix assignment operators generically 
template <typename Matrix, typename ValueType, typename SizeType>
struct crtp_matrix_assign
{
    /// Check whether matrix sizes are compatible or if matrix is 0 by 0 change it to r by c.
    void checked_change_dim(SizeType r, SizeType c)
    {
		Matrix& matrix= static_cast<Matrix&>(*this);
		matrix.check_dim(r, c);
		matrix.change_dim(r, c);
    }

    /// Templated assignment implemented by functor to allow for partial specialization
    // Despite there is only an untemplated assignement and despite the disable_if MSVC whines about ambiguity :-!
    template <typename Source>
    typename boost::disable_if<typename boost::is_same<Matrix, Source>,
			       Matrix&>::type
    operator=(const Source& src)
    {
		return crtp_assign<Source, Matrix>()(src, static_cast<Matrix&>(*this));
    }

    template <typename Source>
    Matrix& operator+=(const Source& src)
    {
		return crtp_plus_assign<Source, Matrix>()(src, static_cast<Matrix&>(*this));
    }
    
    template <typename Source>
    Matrix& operator-=(const Source& src)
    {
		return crtp_minus_assign<Source, Matrix>()(src, static_cast<Matrix&>(*this));
    }
    
    /// Scale matrix (in place) with scalar value or other matrix
    template <typename Factor>
    Matrix& operator*=(const Factor& alpha)
    {
		right_scale_inplace(static_cast<Matrix&>(*this), alpha);
		return static_cast<Matrix&>(*this);
    }
    
    /// Divide matrix (in place) by scalar value
    // added by Hui Li
    template <typename Factor>
    Matrix& operator/=(const Factor& alpha)
    {
		divide_by_inplace(static_cast<Matrix&>(*this), alpha);
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

    // Compiler error (later) if no sub_matrix function available
    operations::range_bracket_proxy<Matrix, const Matrix&, const Matrix>
    operator[] (irange row_range) const
    {
	return operations::range_bracket_proxy<Matrix, const Matrix&, const Matrix>(static_cast<const Matrix&>(*this), row_range);
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

    // Compiler error (later) if no sub_matrix function available
    operations::range_bracket_proxy<Matrix, const Matrix&, const Matrix>
    operator[] (irange row_range) const
    {
	return operations::range_bracket_proxy<Matrix, const Matrix&, const Matrix>(static_cast<const Matrix&>(*this), row_range);
    }

    // Compiler error (later) if no sub_matrix function available
    operations::range_bracket_proxy<Matrix, Matrix&, Matrix>
    operator[] (irange row_range)
    {
	return operations::range_bracket_proxy<Matrix, Matrix&, Matrix>(static_cast<Matrix&>(*this), row_range);
    }
};

template <typename Matrix, typename ValueType, typename SizeType>
struct crtp_matrix_lvalue 
{ 
    // Function must be overwritten by Matrix if m(row, col) does not return a reference
    ValueType& lvalue(SizeType row, SizeType col)
    {
	return static_cast<Matrix&>(*this)(row, col);
    }   
};

template <typename Matrix, typename ValueType, typename SizeType>
struct const_crtp_base_matrix
    : public const_crtp_matrix_bracket<Matrix, ValueType, SizeType>
{};

template <typename Matrix, typename ValueType, typename SizeType>
struct mutable_crtp_base_matrix 
    : public crtp_matrix_bracket<Matrix, ValueType, SizeType>,
      public crtp_matrix_assign<Matrix, ValueType, SizeType>
{};

template <typename Matrix, typename ValueType, typename SizeType>
struct crtp_base_matrix 
    : boost::mpl::if_<boost::is_const<Matrix>,
	                  const_crtp_base_matrix<Matrix, ValueType, SizeType>,
		              mutable_crtp_base_matrix<Matrix, ValueType, SizeType>
                     >::type
{};


}} // namespace mtl::detail

#endif // MTL_CRTP_BASE_MATRIX_INCLUDE
