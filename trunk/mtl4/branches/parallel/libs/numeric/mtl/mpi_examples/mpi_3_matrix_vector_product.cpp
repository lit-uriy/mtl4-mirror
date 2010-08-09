// Filename: mpi_3_matrix_vector_product.cpp

#include <iostream>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    boost::mpi::environment      env(argc, argv);

    typedef mtl::matrix::distributed<mtl::compressed2D<float> >  matrix_type;
    matrix_type A(7, 7);
    A= 4.0;

    mtl::vector::distributed<mtl::dense_vector<float> > u, v(7, 3.0);
    
    u= A * v;

    mtl::par::sout << "The product A * v is " << u << "\n";
    return 0;
}
