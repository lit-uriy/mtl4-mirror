#include <iostream>

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/timer.hpp>
#include <boost/timer.hpp>


const unsigned rep= 100000000; 

template <unsigned BSize, typename Vector>
void run(Vector& u, const Vector& v, Vector& w, Vector&)
{
    using mtl::unroll;

    boost::timer t1; 
    for (unsigned j= 0; j < rep; j++)
	unroll<BSize>(u)= v + w;
    std::cout << "Compute time unroll<" << BSize << ">(u)= v + v is " 
	      << 1000000.0 * t1.elapsed() / double(rep) << " us = "
	      << double(rep) * size(u) / t1.elapsed() / 1e9  << " GFlops.\n";
    // std::cout << "u is " << u << '\n';

}

struct add
{
    template <typename T>
    T operator()(const T& x, const T& y)
    { return x + y; }
};

struct as
{
    template <typename T>
    T& operator()(T& x, const T& y)
    { return x= y; }
};

template <typename T>
void f(T& a)
{
    T& x= a;
    x= "fasdf";
}

int main(int argc, char** argv)
{
    using namespace mtl;
    const unsigned cs= 500;
    unsigned s= cs;
    if (argc > 1) s= atoi(argv[1]);
    
#if 0
    double a[cs] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__))), 
           b[cs] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__))), 
	   c[cs] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));
#else
    double *a= new double[s], 
           *b= new double[s], 
	   *c= new double[s];
#endif
    // typedef dense_vector<double, vector::parameters<tag::col_major, vector::fixed::dimension<cs>, true > > vec;
    typedef mtl::dense_vector<double> vec;
    vec u(s), v(s), w(s), x(s);
   
    for (unsigned i= 0; i < s; i++) { 
	a[i]= v[i]= double(i);
	b[i]= w[i]= double(2*i + 15);
    }
    boost::timer t; 
    asm("#before loop");
    for (unsigned j= 0; j < rep; j++)
	u= v + w;
    asm("#after loop");
    std::cout << "Compute time u= v + v is " << 1000000.0 * t.elapsed() / double(rep) << " us = "
	      << double(rep) * size(u) / t.elapsed() / 1e9  << " GFlops.\n";


#if 1
    //double *&ar= a;
    double (&ar)[cs] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)))= reinterpret_cast<double (&)[cs]>(v[0]); // perverser cast auf array ref
    double (&br)[cs] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)))= reinterpret_cast<double (&)[cs]>(w[0]);
    double (&cr)[cs] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)))= reinterpret_cast<double (&)[cs]>(u[0]);
#endif
    // f(a);

    add adder;
    as  assigner;
    t.restart();
    asm("#before c-loop");
    for (unsigned j= 0; j < rep; j++)
	for (unsigned i= 0; i < cs; i++) {
	    assigner(cr[i], adder(ar[i], br[i]));
	    // assigner(c[i], adder(a[i], b[i]));
	}
    asm("#after c-loop");
    std::cout << "Compute time u= v + v is " << 1000000.0 * t.elapsed() / double(rep) << " us = "
	      << double(rep) * s / t.elapsed() / 1e9  << " GFlops.\n";

#if 0

    run<1>(u, v, w, x);
    run<2>(u, v, w, x);
    run<4>(u, v, w, x);
    run<6>(u, v, w, x);
    run<8>(u, v, w, x);
#endif

    std::cout << "c[0] = " << c[0] << ", u[0] = " << u[0] << '\n';
    c[0]= a[0] + b[0];
    return 0 ;
}

