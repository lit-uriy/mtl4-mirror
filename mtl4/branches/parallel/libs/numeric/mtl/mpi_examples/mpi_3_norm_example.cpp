// Filename: mpi_3_norm_example.cpp

#include <iostream>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    using namespace mtl; using mtl::par::sout; 
    boost::mpi::environment                    env(argc, argv);
    vector::distributed<dense_vector<double> > v(7, 4.0);
    
    sout << "one_norm(v) is " << one_norm(v) << '\n';
    sout << "two_norm(v) is " << two_norm(v) << '\n';
    sout << "infinity_norm(v) is " << infinity_norm(v) << '\n';

    return 0;
}
