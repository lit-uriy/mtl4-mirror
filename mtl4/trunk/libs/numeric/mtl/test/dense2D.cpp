// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/tuple/tuple.hpp>

#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/range_generator.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>
#include <boost/numeric/mtl/operations/raw_copy.hpp>

using namespace mtl;
using namespace std;

struct test_dense2D_exception {};

template <typename T1, typename T2>
void check_same_type(T1, T2)
{
    throw test_dense2D_exception();
}

// If same type we're fine
template <typename T1>
void check_same_type(T1, T1) {}





template <typename Parameters, typename ExpRowComplexity, typename ExpColComplexity>
struct test_dense2D
{
    template <typename Matrix, typename Tag, typename ExpComplexity>
    void two_d_iteration(char const* outer, Matrix & matrix)
    {
	typename traits::row<Matrix>::type   r = row(matrix);
	typename traits::col<Matrix>::type   c = col(matrix);
	typename traits::value<Matrix>::type v = value(matrix);
	typedef typename traits::range_generator<Tag, Matrix>::type        cursor_type;
	typedef typename traits::range_generator<Tag, Matrix>::complexity  complexity;
	cout << outer << complexity() << '\n';
	check_same_type(complexity(), ExpComplexity());
	for (cursor_type cursor = begin<Tag>(matrix), cend = end<Tag>(matrix); cursor != cend; ++cursor) {
	    typedef glas::tags::all_t     inner_tag;
	    typedef typename traits::range_generator<inner_tag, cursor_type>::type icursor_type;
	    for (icursor_type icursor = begin<inner_tag>(cursor), icend = end<inner_tag>(cursor); icursor != icend; ++icursor) 
		cout << "matrix[" << r(*icursor) << ", " << c(*icursor) << "] = " << v(*icursor) << '\n';
	}
    }

    void operator() (double element_1_2)
    {
	typedef dense2D<double, Parameters> matrix_type;
	matrix_type   matrix;
	double        val[] = {1., 2., 3., 4., 5., 6.};
	raw_copy(val, val+6, matrix);

	typename traits::row<matrix_type>::type   r = row(matrix);
	typename traits::col<matrix_type>::type   c = col(matrix);
	typename traits::value<matrix_type>::type v = value(matrix);

	typedef glas::tags::nz_t                                         tag;
	typedef typename traits::range_generator<tag, matrix_type>::type cursor_type;
	for (cursor_type cursor = begin<tag>(matrix), cend = end<tag>(matrix); cursor != cend; ++cursor) {
	    cout << "matrix[" << r(*cursor) << ", " << c(*cursor) << "] = " << v(*cursor) << '\n';
	    if (r(*cursor) == 1 && c(*cursor) == 2 && v(*cursor) != element_1_2) throw test_dense2D_exception();
	}

	two_d_iteration<matrix_type, glas::tags::row_t, ExpRowComplexity>("\nRows: ", matrix);
	two_d_iteration<matrix_type, glas::tags::col_t, ExpColComplexity>("\nColumns: ", matrix);

#if 0
	typedef glas::tags::row_t                                                rtag;
	typedef typename traits::range_generator<rtag, matrix_type>::type        rcursor_type;
	typedef typename traits::range_generator<rtag, matrix_type>::complexity  row_complexity;
	cout << "Rows: " << row_complexity() << '\n';
	check_same_type(row_complexity(), ExpRowComplexity());
	for (rcursor_type cursor = begin<rtag>(matrix), cend = end<rtag>(matrix); cursor != cend; ++cursor) {
	    typedef glas::tags::all_t     ctag;
	    typedef typename traits::range_generator<ctag, rcursor_type>::type ccursor_type;
	    for (ccursor_type ccursor = begin<ctag>(cursor), ccend = end<ctag>(cursor); ccursor != ccend; ++ccursor) 
		cout << "matrix[" << r(*ccursor) << ", " << c(*ccursor) << "] = " << v(*ccursor) << '\n';
	}
	
	cout << '\n';
	typedef glas::tags::col_t                                                ctag;
	typedef typename traits::range_generator<ctag, matrix_type>::type        ccursor_type;
	typedef typename traits::range_generator<ctag, matrix_type>::complexity  col_complexity;
	cout << "Columns: " << col_complexity() << '\n';
	check_same_type(col_complexity(), ExpColComplexity());
	for (ccursor_type cursor = begin<ctag>(matrix), cend = end<ctag>(matrix); cursor != cend; ++cursor) {
	    typedef glas::tags::all_t     rtag;
	    typedef typename traits::range_generator<rtag, ccursor_type>::type rcursor_type;
	    for (rcursor_type rcursor = begin<rtag>(cursor), rcend = end<rtag>(cursor); rcursor != rcend; ++rcursor) 
		cout << "matrix[" << r(*rcursor) << ", " << c(*rcursor) << "] = " << v(*rcursor) << '\n';
	} 
#endif

	cout << '\n';
	typedef transposed_view<matrix_type> trans_matrix_type;
	trans_matrix_type   trans_matrix(matrix);
    
	typename traits::row<trans_matrix_type>::type   tr= row(trans_matrix);
	typename traits::col<trans_matrix_type>::type   tc = col(trans_matrix);
	typename traits::value<trans_matrix_type>::type tv = value(trans_matrix);

	typedef typename traits::range_generator<tag, trans_matrix_type>::type trans_cursor_type;
	for (trans_cursor_type cursor = begin<tag>(trans_matrix), cend = end<tag>(trans_matrix); cursor != cend; ++cursor) {
	    cout << "matrix[" << tr(*cursor) << ", " << tc(*cursor) << "] = " << tv(*cursor) << '\n';
	    if (tr(*cursor) == 2 && tc(*cursor) == 1 && tv(*cursor) != element_1_2) throw test_dense2D_exception();
	}
    }
};

int test_main(int argc, char* argv[])
{
    typedef matrix_parameters<row_major, mtl::index::c_index, fixed::dimensions<2, 3> > parameters1;
    test_dense2D<parameters1, complexity::linear_cached, complexity::linear>()(6.0);

    typedef matrix_parameters<row_major, mtl::index::f_index, fixed::dimensions<2, 3> > parameters2;
    test_dense2D<parameters2, complexity::linear_cached, complexity::linear>()(2.0);

    typedef matrix_parameters<col_major, mtl::index::c_index, fixed::dimensions<2, 3> > parameters3;
    test_dense2D<parameters3, complexity::linear, complexity::linear_cached>()(6.0);

    typedef matrix_parameters<col_major, mtl::index::f_index, fixed::dimensions<2, 3> > parameters4;
    test_dense2D<parameters4, complexity::linear, complexity::linear_cached>()(3.0);

    return 0;
}
