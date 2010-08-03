// Filename: mpi_3_distributed_matrix.cpp

#include <iostream>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    boost::mpi::environment env(argc, argv);

    typedef mtl::matrix::distributed<mtl::compressed2D<float> >  matrix_type;
    matrix_type A(8, 8);
    A= 3.0;

    mtl::par::sout << "The matrix A is\n" << A << "\n";
    return 0;
}
