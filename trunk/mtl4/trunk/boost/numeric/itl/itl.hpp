#include <iostream>
#include <complex>
#include <string>



namespace itl {


  template <class Real>
  class basic_iteration {
  public:

  
    typedef Real real;

  
    template <class Vector>
    basic_iteration(const Vector& b, int max_iter_, Real t, Real a = Real(0))
      : error(0), i(0), normb_(std::abs(two_norm(b))), 
	       max_iter(max_iter_), rtol_(t), atol_(a) { }
  
    basic_iteration(Real nb, int max_iter_, Real t, Real a = Real(0))
      : error(0), i(0), normb_(nb), max_iter(max_iter_), rtol_(t), atol_(a) {}

  
    template <class Vector>
    bool finished(const Vector& r) {
      Real normr_ = std::abs(two_norm(r)); 
      if (converged(normr_))
	return true;
      else if (i < max_iter)
	return false;
      else {
	error = 1;
	return true;
      }
    }

  
    bool finished(const Real& r) {
      if (converged(r))
	return true;
      else if (i < max_iter)
	return false;
      else {
	error = 1;
	return true;
      }
    }

    template <typename T>
    bool finished(const std::complex<T>& r) { 
      if (converged(std::abs(r)))
	return true;
      else if (i < max_iter)
	return false;
      else {
	error = 1;
	return true;
      }
    }

    inline bool converged(const Real& r) {
      if (normb_ == 0)
	return r < atol_;  // ignore relative tolerance if |b| is zero
      resid_ = r / normb_;
      return (resid_ <= rtol_ || r < atol_); // relative or absolute tolerance.
    }

    inline void operator++() { ++i; }
  
    inline bool first() { return i == 0; }
  
    inline int error_code() { return error; }
  
    inline int iterations() { return i; }
  
    inline Real resid() { return resid_ * normb_; }
  
    inline Real normb() const { return normb_; }
  
    inline Real tol() { return rtol_; }
    inline Real atol() { return atol_; } 
  
    inline void fail(int err_code) { error = err_code; }
  
    inline void fail(int err_code, const std::string& msg)
    { error = err_code; err_msg = msg; }
  
    inline void set(Real v) { normb_ = v; }

  protected:
    int error;
    int i;
    const Real normb_;
    int max_iter;
    Real rtol_;
    Real atol_;
    Real resid_;
    std::string err_msg;
  };


  template <class Real>
  class noisy_iteration : public basic_iteration<Real> {
    typedef basic_iteration<Real> super;
  public:
  
    template <class Vector>
    noisy_iteration(const Vector& b, int max_iter_, 
		    Real tol_, Real atol_ = Real(0))
      : super(b, max_iter_, tol_, atol_) { }

    template <class Vector>
    bool finished(const Vector& r) {
      using std::cout;
      using std::endl;

      Real normr_ = std::abs(two_norm(r)); 
      bool ret;
      if (this->converged(normr_))
	ret = true;
      else if (this->i < this->max_iter)
	ret = false;
      else {
	this->error = 1;
	ret = true;
      }
      cout << "iteration " << this->i << ": resid " 
           << this->resid()
	   << endl;
      return ret;
    }

  
    bool finished(const Real& r) {
      using std::cout;
      using std::endl;

      bool ret;
      if (this->converged(r))
	ret = true;
      else if (this->i < this->max_iter)
	ret = false;
      else {
	this->error = 1;
	ret = true;
      }
      cout << "iteration " << this->i << ": resid " 
           << this->resid()
	   << endl;
      return ret;
    }

    template <typename T>
    bool finished(const std::complex<T>& r) { //for the case of complex
      using std::cout;
      using std::endl;

      bool ret;
      if (this->converged(std::abs(r)))
	ret = true;
      else if (this->ii < this->imax_iter)
	ret = false;
      else {
	this->error = 1;
	ret = true;
      }
     cout << "iteration " << this->i << ": resid " 
           << this->resid() << endl;
      return ret;
    }
  
    int error_code() {
      using std::cout;
      using std::endl;

      cout << "finished! error code = " << this->error << endl;
      cout << this->iterations() << " iterations" << endl;
      cout << this->resid() << " is actual final residual. " << endl
	   << this->resid()/this->normb() << " is actual relative tolerance achieved. "
	   << endl;
      cout << "Relative tol: " << this->rtol_ << "  Absolute tol: " << this->atol_ << endl;
      return this->error;
    }

  };

#if 0
  struct identity_preconditioner {
    identity_preconditioner operator()() const {
      identity_preconditioner p;
      return p;
    }

    identity_preconditioner left() const {
      identity_preconditioner p;
      return p;
    }

    identity_preconditioner right() const {
      identity_preconditioner p;
      return p;
    }

  };

  
  template <class VecX, class VecZ>
  inline void solve(const identity_preconditioner& M, const VecX& x, 
		    const VecZ& z) {
    itl::copy(x, const_cast<VecZ&>(z));
  }
  
  template <class VecX, class VecZ>
  inline void trans_solve(const identity_preconditioner& M, 
			  const VecX& x, const VecZ& z) {
    itl::copy(x, const_cast<VecZ&>(z));
  }

  template <class Preconditioner, class VecX, class VecZ>
  inline void 
  solve(const Preconditioner& M, const VecX& x, const VecZ& z) {
    M.solve(x, const_cast<VecZ&>(z));
  }

  template <class Preconditioner, class VecX, class VecZ>
  inline void 
  trans_solve(const Preconditioner& M, const VecX& x, const VecZ& z) {
    M.trans_solve(x, const_cast<VecZ&>(z));
  }
#endif

}



