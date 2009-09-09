
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
#include <boost/utility.hpp>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>


using namespace std;


double f(double) { /* cout << "double\n"; */ return 1.0; }
complex<double> f(complex<double>) 
{ 
    //cout << "complex\n"; 
    return complex<double>(1.0, -1.0); 
}


int test_main(int argc, char* argv[])
{
    using namespace mtl;
    unsigned size= 4;

    double b;
    dense_vector<double>                    vec(size), vec1(size);
    dense2D<double>                                      dr(size, size), Q(size, size), R(size, size);
    dense2D<complex<double> >                            dz(size, size), Qz(size, size), Rz(size, size);
    dense2D<double, matrix::parameters<col_major> >      dc(size, size);

    
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
    //dr[4][3]=4;
    //dr[2][3]=3;
    std::cout<<"MAtrix=\n"<< dr <<"\n";
    //std::cout<<"Vector="<< vec <<"\n";

    boost::tie(vec1, b) = householder(vec);

    std::cout<<"START--------------"<< dr[0][0] << "\n";

    boost::tie(Q, R)= qr_factors(dr);

    std::cout<<"MAtrix  R=\n"<< R <<"\n";
    std::cout<<"MAtrix  Q=\n"<< Q <<"\n";
    std::cout<<"MAtrix  A=Q*R--outside\n"<< Q*R <<"\n";

    dz[0][0]=complex<double>(1.0, 0.0);
    dz[0][1]=complex<double>(1.0, 0.0);
    dz[0][2]=complex<double>(1,0);
    dz[1][0]=complex<double>(1,0);
    dz[1][1]=complex<double>(-1,0);
    dz[1][2]=complex<double>(-2,0);
    dz[2][0]=complex<double>(1,0);
    dz[2][1]=complex<double>(-2,0);
    dz[2][2]=complex<double>(1,0);
    dz[3][3]=complex<double>(-10,0);
    std::cout<<"MAtrix complex=\n"<< dz <<"\n";

    // std::cout<<"START-----complex---------"<< dz[0][0] << "\n";
    //
    // Qz= qr_zerl(dz).first;
    // Rz= qr_zerl(dz).second;
    // std::cout<<"MAtrix  R="<< Rz <<"\n";
    // std::cout<<"MAtrix  Q="<< Qz <<"\n";
    // std::cout<<"MAtrix  A=Q*R--outside"<< Qz*Rz <<"\n";

    return 0;
}

