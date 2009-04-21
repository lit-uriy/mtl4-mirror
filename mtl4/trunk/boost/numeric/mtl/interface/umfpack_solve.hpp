// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MATRIX_UMFPACK_SOLVE_INCLUDE
#define MTL_MATRIX_UMFPACK_SOLVE_INCLUDE

#ifdef MTL_HAS_UMFPACK

#include <cassert>
#include <boost/type_traits.hpp>
#include <boost/numeric/mtl/matrix/compressed2D.hpp>
#include <boost/numeric/mtl/matrix/parameter.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/make_copy_or_reference.hpp>
#include <boost/numeric/mtl/operation/merge_complex_vector.hpp>
#include <boost/numeric/mtl/operation/split_complex_vector.hpp>
#include <umfpack.h>

namespace mtl { namespace matrix {

    namespace umfpack {

	// conversion for value_type needed if not double or complex<double> (where possible)
	template <typename Value> struct value {};
	template<> struct value<double> { typedef double type; };
	template<> struct value<float> { typedef double type; };
	template<> struct value<std::complex<double> > { typedef std::complex<double> type; };
	template<> struct value<std::complex<float> > { typedef std::complex<double> type; };

	template <typename Matrix, typename Value, typename Orientation> 
	struct matrix_copy {};

	// If arbitrary compressed matrix -> copy
	template <typename Value, typename Parameters, typename Orientation>
	struct matrix_copy<compressed2D<Value, Parameters>, Value, Orientation>
	{
	    typedef typename value<Value>::type                      value_type;
	    typedef compressed2D<value_type, parameters<col_major> > matrix_type;

	    matrix_copy(const compressed2D<Value, Parameters>& A) : matrix(A) {}

	    matrix_type matrix;
	};

	// CCS matrices with right value_type and size_type are not copied:

	template <typename Parameters>
	struct matrix_copy<compressed2D<double, Parameters>, double, col_major>
	{
	    typedef double                              value_type;
	    typedef compressed2D<double, Parameters>    matrix_type;
	    
	    matrix_copy(const matrix_type& A) : matrix(A) {}

	    const matrix_type& matrix;
	};

	template <typename Parameters>
	struct matrix_copy<compressed2D<std::complex<double>, Parameters>, std::complex<double>, col_major>
	{
	    typedef std::complex<double>                              value_type;
	    typedef compressed2D<std::complex<double>, Parameters>    matrix_type;
	    
	    matrix_copy(const matrix_type& A) : matrix(A) {}

	    const matrix_type& matrix;
	};

	struct error : public domain_error
	{
	    error(const char *s, int code) : domain_error(s), code(code) {}
	    int code;
	};

	inline void check(int res, const char *s)
	{
	    MTL_THROW_IF(res != UMFPACK_OK, error(s, res));
	}

	/// Class for repeated Umfpack solutions
	/** Keeps symbolic and numeric preprocessing. Numeric part can be updated. 
	    Only defined for compressed matrices. **/
	template <typename T> class solver {};

	template <>
	class solver<compressed2D<double, parameters<col_major> > >
	{
	    typedef double                    value_type;
	    typedef parameters<col_major>     Parameters;
	public:
	    explicit solver(const compressed2D<double, parameters<col_major> >& A) 
		: A(A), n(num_rows(A)), Ap(reinterpret_cast<const int*>(A.address_major())), 
		  Ai(reinterpret_cast<const int*>(A.address_minor())), Ax(A.address_data())
	    {
		MTL_THROW_IF(num_rows(A) != num_cols(A), matrix_not_square());
		check(umfpack_di_symbolic(n, n, Ap, Ai, Ax, &Symbolic, Control, Info), "Error in di_symbolic");
		check(umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, Control, Info), "Error in di_numeric");
	    }

	    ~solver()
	    {
		umfpack_di_free_numeric(&Numeric);
		umfpack_di_free_symbolic(&Symbolic);
	    }

	    /// Update numeric part, for matrices that kept the sparsity and changed the values
	    void update_numeric()
	    {
		umfpack_di_free_numeric(&Numeric);
		check(umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, Control, Info), "Error in di_numeric");
	    }

	    /// Update symbolic and numeric part
	    void update()
	    {
		umfpack_di_free_symbolic(&Symbolic);
		check(umfpack_di_symbolic(n, n, Ap, Ai, Ax, &Symbolic, Control, Info), "Error in di_symbolic");
		update_numeric();
	    }

	    /// Solve double system
	    template <typename VectorX, typename VectorB>
	    int operator()(VectorX& x, const VectorB& b)
	    {
		MTL_THROW_IF(num_rows(A) != size(x) || num_rows(A) != size(b), incompatible_size());
		make_in_out_copy_or_reference<dense_vector<value_type>, VectorX> xx(x);
		make_in_copy_or_reference<dense_vector<value_type>, VectorB> bb(b);
		check(umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, &xx.value[0], &bb.value[0], Numeric, Control, Info), "Error in di_numeric");
		return UMFPACK_OK;
	    }

	private:
	    const compressed2D<double, parameters<col_major> >& A;
	    const int      n, *Ap, *Ai;
	    const double   *Ax;
	    double         Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];
	    void           *Symbolic, *Numeric;
	};

	template <>
	class solver<compressed2D<std::complex<double>, parameters<col_major> > >
	{
	    typedef std::complex<double>                    value_type;
	    typedef parameters<col_major>                   Parameters;
	public:
	    explicit solver(const compressed2D<value_type, Parameters>& A) 
		: A(A), n(num_rows(A)), Ap(reinterpret_cast<const int*>(A.address_major())), 
		  Ai(reinterpret_cast<const int*>(A.address_minor()))
	    {
		MTL_THROW_IF(num_rows(A) != num_cols(A), matrix_not_square());
		split_complex_vector(A.data, Ax, Az);
		check(umfpack_zi_symbolic(n, n, Ap, Ai, &Ax[0], &Az[0], &Symbolic, Control, Info), "Error in zi_symbolic");
		check(umfpack_zi_numeric(Ap, Ai, &Ax[0], &Az[0], Symbolic, &Numeric, Control, Info), "Error in zi_numeric");
	    }

	    ~solver()
	    {
		umfpack_zi_free_numeric(&Numeric);
		umfpack_zi_free_symbolic(&Symbolic);
	    }

	    /// Update numeric part, for matrices that kept the sparsity and changed the values
	    void update_numeric()
	    {
		umfpack_zi_free_numeric(&Numeric);
		check(umfpack_zi_numeric(Ap, Ai, &Ax[0], &Az[0], Symbolic, &Numeric, Control, Info), "Error in zi_numeric");
	    }

	    /// Update symbolic and numeric part
	    void update()
	    {
		umfpack_zi_free_symbolic(&Symbolic);
		check(umfpack_zi_symbolic(n, n, Ap, Ai, &Ax[0], &Az[0], &Symbolic, Control, Info), "Error in zi_symbolic");
	    }

	    /// Solve complex system
	    template <typename VectorX, typename VectorB>
	    int operator()(VectorX& x, const VectorB& b)
	    {
		MTL_THROW_IF(num_rows(A) != size(x) || num_rows(A) != size(b), incompatible_size());
		dense_vector<double> Xx(size(x)), Xz(size(x)), Bx, Bz;
		split_complex_vector(b, Bx, Bz);
		check(umfpack_zi_solve(UMFPACK_A, Ap, Ai, &Ax[0], &Az[0], &Xx[0], &Xz[0], &Bx[0], &Bz[0], Numeric, Control, Info), 
		      "Error in zi_solve");
		merge_complex_vector(Xx, Xz, x);
		return UMFPACK_OK;
	    }

	private:
	    const compressed2D<value_type, Parameters>& A;
	    const int      n, *Ap, *Ai;
	    dense_vector<double> Ax, Az;
	    double         Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];
	    void           *Symbolic, *Numeric;
	};

	template <typename Value, typename Parameters>
	class solver<compressed2D<Value, Parameters> >
	  : matrix_copy<compressed2D<Value, Parameters>, Value, typename Parameters::orientation>,
	    public solver<typename matrix_copy<compressed2D<Value, Parameters>, Value, typename Parameters::orientation>::matrix_type >
	{
	    typedef matrix_copy<compressed2D<Value, Parameters>, Value, typename Parameters::orientation> copy_type;
	    typedef solver<typename matrix_copy<compressed2D<Value, Parameters>, Value, typename Parameters::orientation>::matrix_type > solver_type;
	public:
	    explicit solver(const compressed2D<Value, Parameters>& A) 
		: copy_type(A), solver_type(copy_type::matrix)
	    {}
	};
    } // umfpack


template <typename Value, typename Parameters, typename VectorX, typename VectorB>
int umfpack_solve(const compressed2D<Value, Parameters>& A, VectorX& x, const VectorB& b)
{
    umfpack::solver<compressed2D<Value, Parameters> > solver(A);
    return solver(x, b);
}

}} // namespace mtl::matrix

#endif

#endif // MTL_MATRIX_UMFPACK_SOLVE_INCLUDE
