// Filename: mpi_3_vector_expr.cpp

#include <iostream>
#include <complex>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    typedef mtl::vector::distributed<mtl::dense_vector<float> >  vector_type;
    vector_type  u(8, 3.0), v(8, 2.0), w, x;

    w= u + v;
    mtl::par::sout << "w= u + v is " << w << "\n";

    w-= 2 * v;
    mtl::par::sout << "w-= 2 * v is " << w << "\n";

    x= u + v + 4 * w;
    mtl::par::sout << "x= u + v + 4 * w is " << x << "\n";

    x= dot(v, w) * u + dot(u, w) * v + 4 * w;
    mtl::par::sout << "x= dot(u, w) * u + dot(u, w) * v + 4 * w is " << x << "\n";

    mtl::par::sout << "i * x is " << std::complex<double>(0, 1) * x << "\n";

    return 0;
}
