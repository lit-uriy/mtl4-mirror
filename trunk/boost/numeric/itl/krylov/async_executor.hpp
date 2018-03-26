// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef ITL_ASYNC_EXECUTOR_INCLUDE
#define ITL_ASYNC_EXECUTOR_INCLUDE

#include <thread>

#include <boost/numeric/itl/iteration/interruptible_iteration.hpp>


namespace itl {

template <typename Solver>
class async_executor
{
public:
    async_executor(const Solver& solver) : my_solver(solver), my_iter(), my_thread() {}
    
    /// Solve linear system approximately as specified by \p iter
    template < typename HilbertSpaceX, typename HilbertSpaceB, typename Iteration >
    void start_solve(HilbertSpaceX& x, const HilbertSpaceB& b, Iteration& iter) const
    {
        my_iter.set_iter(iter);
        // my_thread= std::thread(&Solver::solve, &my_solver, b, x, my_iter);
        my_thread= std::thread([this, &x, &b]() { my_solver.solve(x, b, my_iter);});
    }

    /// Solve linear system approximately as specified by \p iter
    template < typename HilbertSpaceX, typename HilbertSpaceB, typename Iteration >
    int solve(HilbertSpaceX& x, const HilbertSpaceB& b, Iteration& iter) const
    {
        start_solve(x, b, iter);
        return wait();
    }

    /// Perform one iteration on linear system
    template < typename HilbertSpaceX, typename HilbertSpaceB >
    int solve(HilbertSpaceX& x, const HilbertSpaceB& b) const
    {
        itl::basic_iteration<double> iter(x, 1, 0, 0);
        return solve(x, b, iter);
    }
    
    int wait()
    {
        my_thread.join();
        return my_iter.error_code();
    }
    
    bool is_finished() const { return !my_thread.joinable(); }
    
    int interrupt() { my_iter.interrupt(); return wait(); }
private:
    Solver                          my_solver;
    mutable interruptible_iteration my_iter;
    mutable std::thread             my_thread;
};

template <typename Solver>
inline async_executor<Solver> make_async_executor(const Solver& solver)
{
    return async_executor<Solver>(solver);
}


} // namespace itl

#endif // ITL_ASYNC_EXECUTOR_INCLUDE
