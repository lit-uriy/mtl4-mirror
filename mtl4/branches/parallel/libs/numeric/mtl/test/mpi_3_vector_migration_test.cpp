// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.


#include <iostream>
#include <boost/test/minimal.hpp>



#if defined(MTL_HAS_PARMETIS) && defined(MTL_HAS_MPI)

#include <map>
#include <utility>
#include <vector>
#include <algorithm>


#define MTL_HAS_STD_OUTPUT_OPERATOR // to print std::vector and std::pair
#include <boost/numeric/mtl/mtl.hpp>

#include <parmetis.h>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>

namespace mpi = boost::mpi;

template <typename Inserter>
struct ins
{
    typedef typename Inserter::size_type  size_type;
    ins(Inserter& i, int start) : i(i), v(start) {}
    void operator()(size_type r) {	i[r] << double(v++);  }
    Inserter& i;
    int       v;
};

template <typename Vector>
inline void cv(const Vector& w, unsigned r, double v)
{
    if (w[r] != v) throw "Wrong value;";
}

template <typename Vector>
void test(Vector& v,  const char* name, int version)
{
    typedef typename mtl::Collection<Vector>::size_type size_type;
    typedef std::pair<size_type, size_type>             entry_type;
    typedef std::vector<entry_type>                     vec_type;

    mtl::par::single_ostream sout;
    mtl::par::multiple_ostream<> mout;

    std::vector<idxtype> part;
    mpi::communicator comm(communicator(v));
    {
	mtl::vector::inserter<Vector> mins(v);
	ins<mtl::vector::inserter<Vector> > i(mins, 10*(comm.rank()+1));
        switch (version) {
          case 1: 
	    switch (comm.rank()) {
	      case 0: i(0); i(1); i(2);  std::cout << "version 1\n"; 
		      part.push_back(1); part.push_back(0); part.push_back(1); break;
    	      case 1: i(3); i(4); 
                      part.push_back(0); part.push_back(0); break;
    	      case 2: i(5); i(6);
		      part.push_back(2); part.push_back(2);
    	    }; break;
          case 2: 
    	    switch (comm.rank()) {
	      case 0: i(0); i(1); i(2); std::cout << "\n\nversion 2\n"; 
		      part.push_back(0); part.push_back(1); part.push_back(1); break;
    	      case 1: i(3); i(4); 
                      part.push_back(1); part.push_back(2); break;
    	      case 2: i(5); i(6);
		      part.push_back(2); part.push_back(0);
    	  }; break;
        }
    }

    sout << "Vector is: " << v << '\n';

    mtl::par::block_migration    migration= parmetis_migration(distribution(v), part);
    Vector w(size(v), migration.new_distribution());
    migrate_vector(v, w, migration);

    sout << "Migrated vector is: " << w << '\n';

    switch (version) {
      case 1: 
	switch (comm.rank()) {
          case 0: cv(w, 0, 11.); cv(w, 1, 20.); cv(w, 2, 21.); break;
	  case 1: cv(w, 3, 10.); cv(w, 4, 12.); break;
	  case 2: cv(w, 5, 30.); cv(w, 6, 31.); 
	}; break;
     case 2: 
       switch (comm.rank()) {
	 case 0: cv(w, 0, 10.); cv(w, 1, 31.); break;
	 case 1: cv(w, 2, 11.); cv(w, 3, 12.); cv(w, 4, 20.); break;
	 case 2: cv(w, 5, 21.); cv(w, 6, 30.);
       }; break;
    }

    Vector u(v, migration);
    sout << "Migrated vector (in constructor) is: " << u << '\n';

    Vector z(u, reverse(migration));
    sout << "Back-migrated vector (in constructor) is: " << z << '\n';
    Vector diff(size(u));
    diff= v - z;
    if (one_norm(diff) != 0.0)
	throw "Wrong back-migration.";
}


int test_main(int argc, char* argv[]) 
{
    using namespace mtl;

    mpi::environment env(argc, argv);
    mpi::communicator world;
    
    if (world.size() != 3) {
	std::cerr << "Example works only for 3 processors!\n";
	env.abort(87);
    }

    vector::distributed<dense_vector<double> > v(7), w(7);

    test(v, "dense_vector<double>", 1);
    test(w, "dense_vector<double>", 2);

    return 0;
}

 
#else 

int test_main(int argc, char* argv[]) 
{
    std::cout << "Test requires the definition of MTL_HAS_PARMETIS (and of course"
	      << " the presence of ParMetis).\n";
    return 0;
}

#endif












