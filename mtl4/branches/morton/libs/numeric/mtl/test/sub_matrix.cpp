// $COPYRIGHT$

#include <iostream>
#include <string>
#include <boost/test/minimal.hpp>
#include <boost/tuple/tuple.hpp>

#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/operations/sub_matrix.hpp>
#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>
#include <boost/numeric/mtl/recursion/base_case_test.hpp>
#include <boost/numeric/mtl/recursion/for_each.hpp>

#include "baseCasesBoost.h"

using namespace mtl;
using namespace std;  

    typedef dense2D<double> matrix_type; 
//typedef morton_dense<double,  0x55555555> matrix_type; 
// using recursion;
recursion::bound_test_static<4>    is_base;
//    recursion::max_dim_test             is_base(16);
//  recursion::matrix_recurator<matrix_type> recurator(matrix);

int  callnum = 0, basehit = 0;
int docholcall=0;
int schurcall=0;
int trischurcall=0;
int trisolvecall=0;

template <typename Recurator>
void print_depth_first(Recurator const& recurator, string str)
{
    cout << "\nRecursion: " << str << endl;
    print_matrix_row_cursor(recurator.get_value());
  
    // for full recursion remove the string length limitation
    if (!recurator.is_leaf()) { // && str.length() < 20) {     
	if (!recurator.north_west_empty())
	    print_depth_first(recurator.north_west(), string("north west of ") + str);
	if (!recurator.south_west_empty())
	    print_depth_first(recurator.south_west(), string("south west of ") + str);
	if (!recurator.north_east_empty())
	    print_depth_first(recurator.north_east(), string("north east of ") + str);
	if (!recurator.south_east_empty())
	    print_depth_first(recurator.south_east(), string("south east of ") + str);
    }
} 


template <typename Recurator, typename BaseCaseTest>
void recursive_print(Recurator const& recurator, string str, BaseCaseTest const& is_base)
{
    if (is_base(recurator)) {
	cout << "\nBase case: " << str << endl;
	print_matrix_row_cursor(recurator.get_value());
    } else {
	recursive_print(recurator.north_west(), string("north west of ") + str, is_base);
	recursive_print(recurator.south_west(), string("south west of ") + str, is_base);
	recursive_print(recurator.north_east(), string("north east of ") + str, is_base);
	recursive_print(recurator.south_east(), string("south east of ") + str, is_base);
    }
} 


template <typename Recurator, typename BaseCaseTest>
void recursive_print_checked(Recurator const& recurator, string str, BaseCaseTest const& is_base)
{
    if (is_base(recurator)) {
	cout << "\nBase case: " << str << endl;
	print_matrix_row_cursor(recurator.get_value());
    } else {
	if (!recurator.north_west_empty())
	    recursive_print_checked(recurator.north_west(), string("north west of ") + str, is_base);
	if (!recurator.south_west_empty())
	    recursive_print_checked(recurator.south_west(), string("south west of ") + str, is_base);
	if (!recurator.north_east_empty())
	    recursive_print_checked(recurator.north_east(), string("north east of ") + str, is_base);
	if (!recurator.south_east_empty())
	    recursive_print_checked(recurator.south_east(), string("south east of ") + str, is_base);
    }
} 

struct print_functor
{
    template <typename Matrix>
    void operator() (Matrix const& matrix) const
    {
	print_matrix_row_cursor(matrix);
	cout << endl;
    }
};

template <typename Matrix>
void test_sub_matrix(Matrix& matrix)
{
    using recursion::for_each;

    print_matrix_row_cursor(matrix);
    
    // recursion::min_dim_test             is_base(2);
    // recursion::undivisible_min_dim_test is_base(2);
    recursion::max_dim_test             is_base(16);
    recursion::matrix_recurator<Matrix> recurator(matrix);
    // print_depth_first(recurator, "");
    recursive_print_checked(recurator, "", is_base);
	 
#if 0
    cout << "\n====================\n"
	 <<   "Same with transposed\n"
	 <<   "====================\n\n";

    transposed_view<Matrix> trans_matrix(matrix);

    print_matrix_row_cursor(trans_matrix); 
    recursion::matrix_recurator< transposed_view<Matrix> > trans_recurator(trans_matrix);
    // print_depth_first(trans_recurator, "");
    recursive_print_checked(trans_recurator, "", is_base);
	 
    cout << "\n=============================\n"
	 <<   "Again with recursive for_each\n"
	 <<   "=============================\n\n";

    recursion::for_each(trans_recurator, print_functor(), is_base);
#endif
}



template <typename Matrix>
void print_matrix(Matrix& matrix){
  register int k, kkk, i, ii, j, jj;
 

  for (i=0 ; i<matrix.num_rows(); i++ )       /* j<=i */
    {
      for(j=0; j<matrix.num_cols();  j++ )
      {
	cout.fill (' '); cout.width (8); cout.precision (5); cout.flags (ios_base::left);
	cout << showpoint <<  matrix[i][j] <<"  "; 
      }
      cout << endl;
    }

  return;
}




template <typename Matrix>
void fill_matrix(Matrix& matrix)
{
    typename traits::row<Matrix>::type                                 row(matrix);
    typename traits::col<Matrix>::type                                 col(matrix);
    typename traits::value<Matrix>::type                               value(matrix);
    typedef  glas::tags::nz_t                                          tag;
    typedef typename traits::range_generator<tag, Matrix>::type        cursor_type;
    
    double x= 1.0;
    /*for (cursor_type cursor = begin<tag>(matrix), cend = end<tag>(matrix); cursor != cend; ++cursor) {
	value(*cursor, x);
	x+= 1.0; 
	}*/
    
    for(int i=0;i<matrix.num_rows();i++)
    {
       for(int j=0;j<=i;j++)
       {
         if(i!=j)
	 {
	   matrix[i][j]=x; matrix[j][i]=x; x=x+1.0; 
	 }
       }
    }
  
   double rowsum;
    for(int i=0;i<matrix.num_rows();i++)
    {
      rowsum=0.0;
       for(int j=0;j<matrix.num_cols();j++)
       {
         if(i!=j)
	 {
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

  if(is_base(E))
    {
      typename Recurator::matrix_type  base_E(E.get_value()), base_W(W.get_value()),
	                               base_N(N.get_value());
      schurBase(base_E, base_W, base_N);
    }
  else
    {
      schurcall++;
      schur(E.north_east(),W.north_west(),N.south_west());
      schur(E.north_east(),W.north_east(),N.south_east());
      schur(E.north_west(),W.north_east(),N.north_east());
      schur(E.north_west(),W.north_west(),N.north_west());

      schur(E.south_west(),W.south_west(),N.north_west());
      schur(E.south_west(),W.south_east(),N.north_east());
      schur(E.south_east(),W.south_east(),N.south_east());
      schur(E.south_east(),W.south_west(),N.south_west());
    }
}

template <typename Recurator>
void triSolve(Recurator S, Recurator N)
{
  if (S.is_empty())
    return;

  if (is_base(S))
    {   // printf(" CALLING triSolveBaseBASE CASE\n");
      typename Recurator::matrix_type  base_S(S.get_value()), base_N(N.get_value());
      triSolveBase(base_S, base_N);
    }
  else
    {
      trisolvecall++;
      triSolve(S.north_west(),N.north_west());
      schur(S.north_east(),S.north_west(),N.south_west());
      triSolve(S.north_east(),N.south_east());

      triSolve(S.south_west(),N.north_west());
      schur(S.south_east(),S.south_west(),N.south_west());
      triSolve(S.south_east(),N.south_east());
    }

}

template <typename Recurator>
void triSchur(Recurator E, Recurator W)
{ 
  if (E.is_empty() || W.is_empty())
    return;

  if(is_base(W))
    {
      // printf(" CALLING triSchurBaseBASE CASE\n");
      typename Recurator::matrix_type  base_E(E.get_value()), base_W(W.get_value());
      triSchurBase(base_E, base_W);
    }
  else
    {
      trischurcall++;
      schur(E.south_west(),W.south_west(),W.north_west());
      schur(E.south_west(),W.south_east(),W.north_east());

      triSchur(E.south_east(),W.south_east());
      triSchur(E.south_east(),W.south_west());
      triSchur(E.north_west(),W.north_east());
      triSchur(E.north_west(),W.north_west());
    }


}
 
template <typename Recurator>
void
doCholesky (Recurator recurator)
{
  if (recurator.is_empty())
    return;

  if (is_base (recurator))
    {
      //stencilnp(recurator); 
      //   printf(" CALLING BASE CASE\n");
      typename Recurator::matrix_type  base_matrix(recurator.get_value());
      doCholeskyBase (base_matrix);
      //zerone(recurator.north_east());
      //  cout <<	"----STENCIL over---------------------------------------------------\n";
        basehit++;
	//show_matrix_rec(recurator);
    }
  else
    {
      docholcall++;
      printf(" CALLING RECURSION for current order:        %d\n",++callnum);

      doCholesky(recurator.north_west());
      //zerone(recurator.north_east());

      triSolve(recurator.south_west(),recurator.north_west());

      triSchur(recurator.south_east(),recurator.south_west());

      doCholesky (recurator.south_east());

    }
}

int test_main(int argc, char* argv[])
{

    cout << "=====================\n"
	 << "Morton-ordered matrix\n"
	 << "=====================\n\n";
   
    matrix_type matrix(32, 32);   
  


    fill_matrix(matrix); 
    // test_sub_matrix(matrix);
     recursion::matrix_recurator<matrix_type> recurator(matrix);
    print_matrix(matrix);
    doCholesky(recurator); 
    cout << "\n=============================\n"
	 <<   "Again with cholesky\n"
	 <<   "=============================\n\n";
    print_matrix(matrix); //cout << "\n\n\n\n\n\n";
    // test_sub_matrix(matrix);
verify_matrix(matrix);
    /* cout << "\n=========================\n"
	 << "Doppler matrix (4x4 base)\n"
	 << "=========================\n\n";

      typedef morton_dense<double,  0x55555553> dmatrix_type;    
    dmatrix_type dmatrix(6, 5);   
    fill_matrix(dmatrix); 
    test_sub_matrix(dmatrix);

    cout << "\n======================\n"
	 << "Row-major dense matrix\n"
	 << "======================\n\n";

    dense2D<double, matrix_parameters<> >   rmatrix(non_fixed::dimensions(6, 5));
    fill_matrix(rmatrix); 
    test_sub_matrix(rmatrix);
 
    cout << "=================================\n"
	 << "Vector-like morton-ordered matrix\n"
	 << "=================================\n\n";

    matrix_type vmatrix(17, 2);   
    fill_matrix(vmatrix); 
    test_sub_matrix(vmatrix);*/

  printf("Rec Calls:  \ndocholcall: %d\n schurcall: %d\n trischurcall:%d \ntrisolvecall:%d\n",  docholcall, schurcall, trischurcall,trisolvecall);
   printf("\nbasecase calls  \ndocholeskyhits :%d\n schurhits :%d\n trischurhits :%d\n trisolvehits:%d\n",  docholeskyhits , schurhits , trischurhits, trisolvehits);

    return 0;
}

