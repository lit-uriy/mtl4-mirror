// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschrÃ¤nkt), www.simunova.com.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.
 
#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/tuple/tuple.hpp>

#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/matrix/coordinate2D.hpp>



int test_main(int, char**)
{
    using namespace mtl;
    typedef mtl::matrix::coordinate2D<double> matrix_type;
    matrix_type   matrix(5,4);
    mtl::dense_vector<double> res(5,0.0), x(4,1.0);
    
    std::cout <<"num_rows=" << matrix.num_rows() << "\n";
    std::cout <<"num_rows=" << num_rows(matrix) << "\n";
    std::cout <<"size=" << size(matrix) << "\n";
    std::cout <<"nnz=" << nnz(matrix) << "\n";
    matrix.push_back(1,1,1.33);
    matrix.push_back(1,2,2.33);
    matrix.push_back(2,1,3.33);
    matrix.push_back(3,1,4.33);
    mtl::dense_vector<unsigned int> rows(matrix.row_index_array()),cols(matrix.column_index_array());
    mtl::dense_vector<double> val0(matrix.value_array());
    for(unsigned int i=0;i<size(rows);i++){
      std::cout<<"row[" <<i <<"]=" <<rows[i] << "\n";
      std::cout<<"col[" <<i <<"]=" <<cols[i] << "\n"; 
      std::cout<<"val[" <<i <<"]=" <<val0[i] << "\n";
    }
    std::cout<<"matrix(3,1)=" << matrix(3,1) << "\n";
    std::cout<<"matrix(1,2)=" << matrix(1,2) << "\n";
    std::cout<<"matrix(2,1)=" << matrix(2,1) << "\n";
//     std::cout<<"matrix(4,4)=" << matrix(4,4) << "\n"; //crashes because col>=4
    matrix.push_back(1,0,5.33);
    matrix.push_back(0,0,6.33);
    
    mtl::dense_vector<unsigned int> rows1(matrix.row_index_array()),cols1(matrix.column_index_array());
    mtl::dense_vector<double> val(matrix.value_array());
    for(unsigned int i=0;i<size(rows1);i++){
      std::cout<<"row[" <<i <<"]=" <<rows1[i] << " , col[" <<i <<"]=" <<cols1[i] << " , val[" <<i <<"]=" <<val[i] << "\n";
    }
    matrix.sort();
    mtl::dense_vector<unsigned int> rows2(matrix.row_index_array()),cols2(matrix.column_index_array());
    mtl::dense_vector<double> val2(matrix.value_array());
    
    std::cout<< "SORTING\n";
    for(uint i=0;i<size(rows2);i++){
      std::cout<<"row[" <<i <<"]=" <<rows2[i] << " , col[" <<i <<"]=" <<cols2[i] << " , val[" <<i <<"]=" <<val2[i] << "\n";
    }
   
   
//      matrix.insert(2,2,33.3,5); // cheap inserter... cheaper than push_back  position muss known
//      matrix_type* L =	new matrix_type(5, 5);
//      std::vector< matrix_type*> lower_matrices;
//      lower_matrices.push_back(L);
//        std::cout <<"matrix=" << matrix << "\n";
    std::cout<<"x=" << x<<"\n";
    std::cout<<"res=" << res<<"\n";
    res = matrix * x ;
    std::cout<<"res=" << res<<"\n";
     
    
    matrix_type A(5,5,9);
    A.insert(1,2,13.3,0);
    A.insert(2,2,23.3,1);
    A.insert(2,3,33.3,2);
    A.insert(2,4,43.3,3);
    A.insert(0,4,53.3,4);
    A.insert(1,4,63.3,5);
    A.insert(3,0,73.3,6);
    A.compress();
//     A.insert(4,4,33.3,7); //should fail. because pos>7
    mtl::dense_vector<unsigned int> rows3(A.row_index_array()),cols3(A.column_index_array());
    mtl::dense_vector<double> val3(A.value_array());
    std::cout<<"OK\n";
    for(unsigned int i=0;i<size(rows3);i++){
      std::cout<<"row[" <<i <<"]=" <<rows3[i] << " , col[" <<i <<"]=" <<cols3[i] << " , val[" <<i <<"]=" <<val3[i] << "\n";
    }
    A.print();
    mtl::matrix::compressed2D<double> C(crs(A));
    
    std::cout<< "C=\n"<< C << "\n";
//     C= crs(A);
    
    
    return 0;
}
