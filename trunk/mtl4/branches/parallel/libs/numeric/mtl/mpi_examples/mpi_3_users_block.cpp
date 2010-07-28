// Filename: mpi_3_distributed_vector_update.cpp

#include <iostream>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    typedef mtl::vector::distributed<mtl::dense_vector<float> >  vector_type;
    std::vector<std::size_t> block;
    block.push_back(0); block.push_back(2); block.push_back(7); block.push_back(8); 

    mtl::par::block_distribution dist(block);
    vector_type  v(8, dist);
    
    {
	mtl::vector::inserter<vector_type> ins(v);
	

	for (unsigned i= 0; i < size(v); ++i)
	    if (i % world.size() == world.rank())
		ins[i] << float(world.rank());
	    else
		ins[i] << 0.1;
    }

    std::cout << "I am proc. " << world.rank() << " and my local part of v is " << local(v) << '\n';

    return 0;
}
