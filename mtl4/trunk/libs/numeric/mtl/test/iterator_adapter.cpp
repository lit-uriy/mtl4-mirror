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

//template <typename PropertyMap, typename Cursor, typename ValueType>






template <typename Adapter>
struct adapter_operators
{
    Adapter& operator++() 
    {
	Adapter& me = static_cast<Adapter&>(*this);
	++me.cursor;
	return me;
    }

    Adapter& operator++(int) 
    {
	Adapter& me = static_cast<Adapter&>(*this);
	Adapter  tmp(me);
	++me.cursor;
	return tmp;
    }
    
    bool operator==(Adapter const& x) const
    {
	Adapter const& me = static_cast<Adapter const&>(*this);

	// Compare addresses of property maps
	return &me.map == &x.map && me.cursor == x.cursor;

	// Certainly better they provide comparison
	// return me.map == x.map && me.cursor == x.cursor; 
    }

    bool operator!=(Adapter const& x) const
    {
	return !operator==(x);
	//Adapter const& me = static_cast<Adapter const&>(*this);
	//return !(me == x);
    }
};


template <typename PropertyMap, typename Cursor, typename ValueType>
struct const_iterator_proxy
{
    const_iterator_proxy(PropertyMap const& map, Cursor cursor) 
	: map(map), cursor(cursor) {}

    operator ValueType() const
    {
	return map(*cursor);
    }

    PropertyMap const&     map;
    Cursor                 cursor;
};

template <typename PropertyMap, typename Cursor, typename ValueType>
struct iterator_proxy
{
    typedef iterator_proxy                    self;

    iterator_proxy(PropertyMap& map, Cursor cursor) 
	: map(map), cursor(cursor) {}

    operator ValueType() const
    {
	return map(*cursor);
    }

    self& operator=(ValueType const& value)
    {
	map(*cursor, value);
	return *this;
    }

    PropertyMap&           map;
    Cursor                 cursor;
};


template <typename PropertyMap, typename Cursor, typename ValueType>
struct const_iterator_adapter
    : public adapter_operators< const_iterator_adapter<PropertyMap, Cursor, ValueType> >
//: public iterator_adapter<PropertyMap, Cursor, ValueType>
{
    // typedef iterator_adapter<PropertyMap, Cursor, ValueType>         base;
    typedef const_iterator_adapter                                   self;
    typedef const_iterator_proxy<PropertyMap, Cursor, ValueType>     proxy;

    const_iterator_adapter(PropertyMap const& map, Cursor cursor) 
	: map(map), cursor(cursor) {}

    proxy operator*()
    {
	return proxy(map, cursor);
    }

    PropertyMap const&     map;
    Cursor                 cursor;
};


template <typename PropertyMap, typename Cursor, typename ValueType>
struct iterator_adapter
    : public adapter_operators< iterator_adapter<PropertyMap, Cursor, ValueType> >
{
    typedef iterator_adapter                                 self;
    typedef iterator_proxy<PropertyMap, Cursor, ValueType>   proxy;

    iterator_adapter(PropertyMap& map, Cursor cursor) : map(map), cursor(cursor) {}

#if 0
    self& operator++() 
    {
	++cursor;
	return *this;
    }

    self& operator++(int) 
    {
	self    tmp(*this);
	++cursor;
	return tmp;
    }
    
    bool operator==(self const& x) const
    {
	return map == x.map && cursor == x.cursor;
    }

    bool operator!=(self const& x) const
    {
	return !(*this == x);
    }    
#endif

    proxy operator*()
    {
	return proxy(map, cursor);
    }

    PropertyMap&     map;
    Cursor           cursor;
};


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
    void two_d_iteration(char const* outer, Matrix & matrix, Tag, ExpComplexity)
    {
	typename traits::row<Matrix>::type                         r = row(matrix);
	typename traits::col<Matrix>::type                         c = col(matrix);
	typename traits::value<Matrix>::type                       v = value(matrix);
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

    template <typename Matrix>
    void one_d_iteration(char const* name, Matrix & matrix, size_t check_row, size_t check_col, double check)
    {
	typename traits::row<Matrix>::type                         r = row(matrix);
	typename traits::col<Matrix>::type                         c = col(matrix);
	typename traits::value<Matrix>::type                       v = value(matrix);
	typedef  glas::tags::nz_t                                  tag;
	typedef typename traits::range_generator<tag, Matrix>::type        cursor_type;
	typedef typename traits::range_generator<tag, Matrix>::complexity  complexity;

	cout << name << "\nElements: " << complexity() << '\n';

	for (cursor_type cursor = begin<tag>(matrix), cend = end<tag>(matrix); cursor != cend; ++cursor) {
	    cout << "matrix[" << r(*cursor) << ", " << c(*cursor) << "] = " << v(*cursor) << '\n';
	    if (r(*cursor) == check_row && c(*cursor) == check_col && v(*cursor) != check) throw test_dense2D_exception();
	}

	typedef const_iterator_adapter<typename traits::row<Matrix>::type, cursor_type, double> row_adapter_type;
	row_adapter_type row_iter(r, begin<tag>(matrix)), row_end(r, end<tag>(matrix));

	typedef const_iterator_adapter<typename traits::col<Matrix>::type, cursor_type, double> col_adapter_type;
	col_adapter_type col_iter(c, begin<tag>(matrix));

	typedef iterator_adapter<typename traits::value<Matrix>::type, cursor_type, double> value_adapter_type;
	value_adapter_type value_iter(v, begin<tag>(matrix));

	cout << "\nSame with iterator adapter.\n";
	for (; row_iter != row_end; ++row_iter, ++col_iter, ++value_iter) {
	    cout << "matrix[" << *row_iter << ", " << *col_iter << "] = " << *value_iter << '\n';
	}

	

    }
    
    void operator() (double element_1_2)
    {
	typedef dense2D<double, Parameters> matrix_type;
	matrix_type   matrix;
	double        val[] = {1., 2., 3., 4., 5., 6.};
	raw_copy(val, val+6, matrix);

	one_d_iteration("\nMatrix", matrix, 1, 2, element_1_2);
	two_d_iteration("\nRows: ", matrix, glas::tags::row_t(), ExpRowComplexity());
	two_d_iteration("\nColumns: ", matrix, glas::tags::col_t(), ExpColComplexity());

	transposed_view<matrix_type> trans_matrix(matrix);
	one_d_iteration("\nTransposed matrix", trans_matrix, 2, 1, element_1_2);
	two_d_iteration("\nRows: ", trans_matrix, glas::tags::row_t(), ExpColComplexity());
	two_d_iteration("\nColumns: ", trans_matrix, glas::tags::col_t(), ExpRowComplexity());
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
