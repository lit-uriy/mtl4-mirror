/*
 *  cholesky_boost.cpp
 *  Jan07 
 *  Created by  Adwait Joshi
 *  Project: Arcee
 *
 *  Copyright (c) 2007 The Trustees of Indiana University. All rights reserved.
 *
 */

#include <iostream>
#include <string>
#include <boost/test/minimal.hpp>
#include <boost/tuple/tuple.hpp>

#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/matrix/morton_dense.hpp>
#include <boost/numeric/mtl/matrix/transposed_view.hpp>
#include <boost/numeric/mtl/matrix/parameter.hpp>
#include <boost/numeric/mtl/operation/print_matrix.hpp>
#include <boost/numeric/mtl/operation/sub_matrix.hpp>
#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>
#include <boost/numeric/mtl/recursion/base_case_test.hpp>
#include <boost/numeric/mtl/recursion/for_each.hpp>

//#include "base_Cases_Boost.h"
#include "base_Cases_Boost_new.h"

using namespace mtl;
using namespace std;  

typedef dense2D<double> matrix_type; 
//typedef morton_dense<double,  0x55555555> matrix_type; 
const int order=4200 ; 
const int basecasesize =32;
recursion::bound_test_static<basecasesize>    is_base;


int  callnum = 0, basehit = 0;
int docholcall=0;
int schurcall=0;
int trischurcall=0;
int trisolvecall=0;


template <typename Matrix>
void print_matrix(Matrix& matrix){ 
	for (int i=0 ; i<matrix.num_rows(); i++ ){
		for(int j=0; j<matrix.num_cols();  j++ ){
		   cout.fill (' '); cout.width (8); cout.precision (5); cout.flags (ios_base::left);
		   cout << showpoint <<  matrix[i][j] <<"  ";
		}
		cout << endl;
	}
	return;
}




template <typename Matrix>
void fill_matrix(Matrix& matrix){
    typename traits::row<Matrix>::type                                 row(matrix);
    typename traits::col<Matrix>::type                                 col(matrix);
    typename traits::value<Matrix>::type                               value(matrix);
    typedef  glas::tag::nz                                          tag;
    typedef typename traits::range_generator<tag, Matrix>::type        cursor_type;
    
    double x= 1.0;      
    for(int i=0;i<matrix.num_rows();i++) {
       for(int j=0;j<=i;j++){
         if(i!=j){
	          matrix[i][j]=x; matrix[j][i]=x; 
	          x=x+1.0; 
	       }
       }
    }
  
    double rowsum;
    for(int i=0;i<matrix.num_rows();i++){
       rowsum=0.0;
       for(int j=0;j<matrix.num_cols();j++){
         if(i!=j){
	          rowsum += matrix[i][j]; 
	       }
       }
       matrix[i][i]=rowsum*2;
    }       
}


template <typename Recurator>
void schur(Recurator E, Recurator W, Recurator N)
{
  if (E.is_empty() || W.is_empty() || N.is_empty())
    return;

  if(is_base(E)){
     typename Recurator::matrix_type  base_E(E.get_value()), base_W(W.get_value()),base_N(N.get_value());
     schur_base(base_E, base_W, base_N);
  }
  else{
    schur(     E.north_east(),W.north_west()     ,N.south_west()     );
    schur(     E.north_east(),     W.north_east(),     N.south_east());
    
    schur(E.north_west()     ,     W.north_east(),     N.north_east());
    schur(E.north_west()     ,W.north_west()     ,N.north_west()     );
    
    schur(E.south_west()     ,W.south_west()     ,N.north_west()     );
    schur(E.south_west()     ,     W.south_east(),     N.north_east());
    
    schur(     E.south_east(),     W.south_east(),     N.south_east());
    schur(     E.south_east(),W.south_west()     ,N.south_west()     );
  }
}





template <typename Recurator>
void tri_solve(Recurator S, Recurator N)
{
  if (S.is_empty())
    return;

  if(is_base(S)){   
     typename Recurator::matrix_type  base_S(S.get_value()), base_N(N.get_value());
     tri_solve_base(base_S, base_N);
  }
  else{ 
  tri_solve(S.north_west()     ,N.north_west()     );
  	 
      schur(     S.north_east(),S.north_west()     ,N.south_west()     );
     
  tri_solve(     S.north_east(),     N.south_east());

  tri_solve(S.south_west()     ,N.north_west()     );
     
      schur(     S.south_east(),S.south_west()     ,N.south_west()     );
     
  tri_solve(     S.south_east(),     N.south_east());
  }
}





template <typename Recurator>
void tri_schur(Recurator E, Recurator W)
{ 
  if (E.is_empty() || W.is_empty())
    return;

  if(is_base(W)){
     typename Recurator::matrix_type  base_E(E.get_value()), base_W(W.get_value());
     tri_schur_base(base_E, base_W);
  }
  else{ 
      schur(     E.south_west(),     W.south_west(),    W.north_west());

      schur(     E.south_west(),     W.south_east(),    W.north_east());
	
  tri_schur(E.south_east()     ,     W.south_east());

  tri_schur(E.south_east()     ,W.south_west()     );

  tri_schur(     E.north_west(),     W.north_east());

  tri_schur(     E.north_west(),W.north_west()     );
  }
}

 
template <typename Recurator>
void
do_cholesky (Recurator recurator)
{
  if (recurator.is_empty())
    return;

  if (is_base (recurator)){    
      typename Recurator::matrix_type  base_matrix(recurator.get_value());
      do_cholesky_base (base_matrix);      
  }
  else{

    do_cholesky(recurator.north_west()     );
      
      tri_solve(recurator.south_west()     ,recurator.north_west()     );

      tri_schur(     recurator.south_east(),recurator.south_west()     );

    do_cholesky(     recurator.south_east());

  }
}

int test_main(int argc, char* argv[])
{

  //    cout << "=====================\n" << "Morton-ordered matrix\n" << "=====================\n\n";
  matrix_type matrix(order,order);   
  time_t starttime,endtime;
  struct tm *timeinfo;
  
  time (&starttime);
  timeinfo = localtime (&starttime);
  
  printf("----------order = %d      Basecase = %d  -------------------->Load start: %s",
	 order, basecasesize, asctime (timeinfo));
  
  
  fill_matrix(matrix); 
  // test_sub_matrix(matrix);
  recursion::matrix_recurator<matrix_type> recurator(matrix);
  // print_matrix(matrix);
  time (&starttime);
  timeinfo = localtime (&starttime);
  printf("----------order = %d      Basecase = %d  -------------------->Start date and time are: %s",
     order, basecasesize, asctime (timeinfo));
					  
  do_cholesky(recurator); 
  
  
  //    cout << "\n=============================\n"	 <<   "Again with cholesky\n"	 <<   "=============================\n\n";
  
  time (&endtime);
  timeinfo = localtime (&endtime);
  printf("----------order = %d      Basecase = %d  ------------------->End date and time are: %s",
     order, basecasesize, asctime (timeinfo));
  //printf ("\nRec calls: %d    Basehits: %d\n", callnum, basehit);
  
  printf
    ("\nTOTAL TIME TAKEN for order = %d      Basecase = %d  : %d  secs\n\n",
     order, basecasesize, endtime - starttime);
  
   //cout << "\n\n\n\n\n\n";
     
 // print_matrix(matrix); 
 /*     cout << "\n=============================\n"	;
   
		for(int i=0 ; i<matrix.num_rows();  i++ ){
			for (int j=i+1; j<matrix.num_cols(); j++ )			 
	    matrix[i][j]=0;
	    }
   
   print_matrix(matrix); 
   cout  <<      "=============================\n\n";
    verify_matrix(matrix);
   cout  <<      "=============================\n\n";
   print_matrix(matrix);*/

     //printf("Rec Calls:\ndocholcall: %d\nschurcall: %d\ntrischurcall:%d\ntrisolvecall:%d\n", 	 docholcall, schurcall, trischurcall,trisolvecall);
     //printf("\nbasecase calls\ndocholhits:%d\nschurhits:%d\ntrischurhits:%d\ntrisolvehits:%d\n",	   docholeskyhits , schurhits , trischurhits, trisolvehits);

    return 0;
}

