// Filename: mpi_3_single_ostream.cpp

#include <iostream>
#include <cstdlib>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    
    srandom(1000 * world.rank());
    long int r1= random(), r2= random();

    mtl::par::single_ostream sout;
    mtl::par::rank_ostream   rout;

    sout << "Example with output on each process.\n";
    rout << "My random values are " << r1 << " and " << r2 << ".\n";
    sout << "End of output examples.\n";

    return 0;
}
