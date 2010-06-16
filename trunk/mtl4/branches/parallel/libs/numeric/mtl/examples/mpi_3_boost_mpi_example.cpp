// Filename: mpi_3_boost_mpi_example.cpp

#include <iostream>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    mtl::dense_vector<float>  v(3);

    // set vector on processor 0 and send it to others
    if (world.rank() == 0) {
	v= 3.0, 4.0, 5.5;
	for (int i= 1; i < world.size(); i++)
	    world.send(i, 0, v);
    } else {
	world.recv(0, 0, v);
	std::cout << "Hello world, I am proc. " << world.rank() << " and proc. 0 set v to " << v << '\n';
    }

    return 0;
}
