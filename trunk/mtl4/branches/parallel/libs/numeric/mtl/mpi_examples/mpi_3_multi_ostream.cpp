// Filename: mpi_3_multi_ostream.cpp

#include <iostream>
#include <cstdlib>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    using mtl::par::sout; using mtl::par::rout; 
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    
    srandom(1000 * world.rank());
    long int r1= random(), r2= random();

    sout << "Example with output on each process.\n";
    rout << "My random values are " << r1 << " and " << r2 << ".\n";
    sout << "End of output examples.\n";

    return 0;
}
