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
#include <boost/numeric/mtl/mtl.hpp>


using namespace std;


double f(double) { cout << "double\n"; return 1.0; }
complex<double> f(complex<double>) { cout << "complex\n"; return
complex<double>(1.0, -1.0); }


int test_main(int argc, char* argv[])
{
    using namespace mtl;
    unsigned size= 5;

    dense_vector<double>                    vecr(size), Qr;
    dense2D<double>                         dr(size, size), R(size, size);
    dense2D<complex<double> >               dz(size, size), Qz(size, size);
    dense_vector<complex<double> >          vecz(size);

    dr[0][0]=1;
    dr[0][1]=1;
    dr[0][2]=1;
    dr[1][0]=1;
    dr[1][1]=-1;
    dr[1][2]=-2;
    dr[2][0]=1;
    dr[2][1]=-2;
    dr[2][2]=1;
    dr[3][3]=-10;
    dr[4][4]=22;

    std::cout<<"START------eigenvalue-------double-------"<< endl;
    Qr= eigenvalue_symmetric(dr,22);

    std::cout<<"eigenvalues  ="<< Qr <<"\n";


    // std::cout<<"START------eigenvalue-------complex-------"<< endl;
    //
    // dz[0][0]=complex<double>(1.0, 0.0);
    // dz[0][1]=complex<double>(1.0, 0.0);
    // dz[0][2]=complex<double>(1,0);
    // dz[1][0]=complex<double>(1,0);
    // dz[1][1]=complex<double>(-1,0);
    // dz[1][2]=complex<double>(-2,0);
    // dz[2][0]=complex<double>(1,0);
    // dz[2][1]=complex<double>(-2,0);
    // dz[2][2]=complex<double>(1,0);
    // dz[3][3]=complex<double>(-10,0);
    //
    // //std::cout<<"MAtrix complex="<< dz <<"\n";
    //
    // Qz= eigenvalue(dz,22);
    //
    // //std::cout<<"MAtrix  Q="<< Q <<"\n";
    // for(int i= 0; i < size; i++)
    //     vecz[i]= Qz[i][i];
    // std::cout<<"eigenvalues  ="<< vecz <<"\n";

    return 0;
}



