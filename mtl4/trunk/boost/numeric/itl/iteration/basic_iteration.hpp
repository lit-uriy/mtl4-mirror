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

#ifndef ITL_BASIC_ITERATION_INCLUDE
#define ITL_BASIC_ITERATION_INCLUDE

#include <iostream>
#include <complex>
#include <string>

namespace itl {


template <class Real>
class basic_iteration
{
  public:

    typedef Real real;

    template <class Vector>
    basic_iteration(const Vector& r0, int max_iter_, Real t, Real a = Real(0))
      : error(0), i(0), norm_r0(std::abs(two_norm(r0))),
	max_iter(max_iter_), rtol_(t), atol_(a), is_finished(false), my_quite(false) { }

    basic_iteration(Real nb, int max_iter_, Real t, Real a = Real(0))
      : error(0), i(0), norm_r0(nb), max_iter(max_iter_), rtol_(t), atol_(a), is_finished(false), my_quite(false) {}

    virtual ~basic_iteration() {}

    template <class Vector>
    bool finished(const Vector& r) {
	Real normr_ = two_norm(r);
	if (converged(normr_)) {
	    is_finished= true;
	    return true;
	} else if (i < max_iter)
	    return false;
	else {
	    is_finished= true;
	    error = 1;
	    return true;
	}
    }


    bool finished(const Real& r)
    {
	if (converged(r)) {
	    is_finished= true;
	    return true;
	} else if (i < max_iter)
	    return false;
	else {
	    is_finished= true;
	    error = 1;
	    return true;
	}
    }

    template <typename T>
    bool finished(const std::complex<T>& r) 
    {
	if (converged(std::abs(r))) {
	    is_finished= true;
	    return true;
	} else if (i < max_iter)
	    return false;
	else {
	    is_finished= true;
	    error = 1;
	    return true;
	}
    }

    bool finished() { return is_finished; }

    inline bool converged(const Real& r) 
    {
	if (norm_r0 == 0)
	    return r < atol_;  // ignore relative tolerance if |r0| is zero
	resid_ = r / norm_r0;
	return (resid_ <= rtol_ || r < atol_); // relative or absolute tolerance.
    }

    inline void operator++() { ++i; }

    inline void operator+=(int n) { i+= n; }

    inline bool first() { return i == 0; }

    virtual operator int() { return error; }

    virtual int error_code() { return error; }

    inline int iterations() { return i + 1; }

    inline Real resid() { return resid_ * norm_r0; }

    inline Real normb() const { return norm_r0; }

    inline Real tol() { return rtol_; }
    inline Real atol() { return atol_; }

    inline int fail(int err_code) { error = err_code; return error_code(); }

    inline int fail(int err_code, const std::string& msg)
    { error = err_code; err_msg = msg; return error_code(); }

    inline void set(Real v) { norm_r0 = v; }

    void set_quite(bool q) { my_quite= q; }

    bool is_quite() const { return my_quite; }

  protected:
    int          error, i;
    const Real   norm_r0;
    int          max_iter;
    Real         rtol_, atol_, resid_;
    std::string  err_msg;
    bool         is_finished, my_quite;
};


} // namespace itl

#endif // ITL_BASIC_ITERATION_INCLUDE
