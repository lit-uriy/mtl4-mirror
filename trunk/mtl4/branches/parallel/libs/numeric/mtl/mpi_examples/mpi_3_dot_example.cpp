// Filename: mpi_3_dot_example.cpp

#include <iostream>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    boost::mpi::environment      env(argc, argv);

    mtl::vector::distributed<mtl::dense_vector<double> > u(7, 4.0);
    mtl::vector::distributed<mtl::dense_vector<float> >  v(7, 3.0f);
    
    mtl::par::sout << "dot(u, v) is " << dot(u, v) << '\n';

    return 0;
}
