// Filename: matrix_free_3.cpp (part of MTL4)

#include <iostream>
#include <cassert>
#include <boost/numeric/mtl/mtl.hpp>

template <typename Matrix, typename VectorIn>
struct multiplier
  : mtl::vector::assigner<multiplier<Matrix, VectorIn> >,
    mtl::vector::incrementer<multiplier<Matrix, VectorIn> >,
    mtl::vector::decrementer<multiplier<Matrix, VectorIn> >
{
    multiplier(const Matrix& A, const VectorIn& v) : A(A), v(v) {}

    template <typename VectorOut>
    void assign_to(VectorOut& w) const
    {
	A.mult(v, w, mtl::assign::assign_sum());
    }

    template <typename VectorOut>
    void increment_it(VectorOut& w) const
    {
	A.mult(v, w, mtl::assign::plus_sum());
    }

    template <typename VectorOut>
    void decrement_it(VectorOut& w) const
    {
	A.mult(v, w, mtl::assign::minus_sum());
    }

    const Matrix&   A;
    const VectorIn& v;
};


struct poisson2D_dirichlet
{
    poisson2D_dirichlet(int m, int n) : m(m), n(n) {}

    template <typename VectorIn, typename VectorOut, typename Assign>
    void mult(const VectorIn& v, VectorOut& w, Assign) const
    {
	assert(int(size(v)) == m * n);
	assert(size(v) == size(w));

	// Inner domain
	for (int i= 1; i < m-1; i++)
	    for (int j= 1, k= i * n + j; j < n-1; j++, k++) 
		Assign::apply(w[k], 4 * v[k] - v[k-n] - v[k+n] - v[k-1] - v[k+1]); 
	    
	// Upper border
	for (int j= 1; j < n-1; j++) 
	    Assign::apply(w[j], 4 * v[j] - v[j+n] - v[j-1] - v[j+1]);

	// Lower border
	for (int j= 1, k= (m-1) * n + j; j < n-1; j++, k++) 
	    Assign::apply(w[k], 4 * v[k] - v[k-n] - v[k-1] - v[k+1]); 
	
	// Left border
	for (int i= 1, k= n; i < m-1; i++, k+= n)
	    Assign::apply(w[k], 4 * v[k] - v[k-n] - v[k+n] - v[k+1]); 

	// Right border
	for (int i= 1, k= n+n-1; i < m-1; i++, k+= n)
	    Assign::apply(w[k], 4 * v[k] - v[k-n] - v[k+n] - v[k-1]); 

	// Corners
	Assign::apply(w[0], 4 * v[0] - v[1] - v[n]);
	Assign::apply(w[n-1], 4 * v[n-1] - v[n-2] - v[2*n - 1]);
	Assign::apply(w[(m-1)*n], 4 * v[(m-1)*n] - v[(m-2)*n] - v[(m-1)*n+1]);
	Assign::apply(w[m*n-1], 4 * v[m*n-1] - v[m*n-2] - v[m*n-n-1]);
    }

    template <typename VectorIn>
    multiplier<poisson2D_dirichlet, VectorIn> operator*(const VectorIn& v) const
    {	return multiplier<poisson2D_dirichlet, VectorIn>(*this, v);    }

    int m, n;
};

namespace mtl { namespace ashape {
    template <> struct ashape_aux<poisson2D_dirichlet> 
    {	typedef nonscal type;    };
}}

int main(int, char**)
{
    using namespace std;
    typedef mtl::dense_vector<double> vt;
    
    vt v(20);
    iota(v);
    cout << "v is " << v << endl;

    poisson2D_dirichlet A(4, 5);
    vt  w2(20);
    w2= A * v;
    cout << "A * v is " << w2 << endl;

    w2+= A * v;
    cout << "w2+= A * v is " << w2 << endl;

    w2-= A * v;
    cout << "w2-= A * v is " << w2 << endl;

    return 0;
}
