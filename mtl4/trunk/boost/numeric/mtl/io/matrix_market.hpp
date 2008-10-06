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

#ifndef MTL_IO_MATRIX_MARKET_INCLUDE
#define MTL_IO_MATRIX_MARKET_INCLUDE

#include <string>
#include <fstream>
#include <iostream>

#include <boost/algorithm/string/case_conv.hpp>

#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/utility/string_to_enum.hpp>
#include <boost/numeric/mtl/matrix/inserter.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>

namespace mtl { namespace io {


/// Input file stream for files in matrix market format
class matrix_market_ifstream
{
    class pattern_type {};
    typedef matrix_market_ifstream        self;
  public:
    explicit matrix_market_ifstream(const char* p) : my_stream(p) {}

    template <typename Coll>
    self& operator>>(Coll& c) 
    { 
	return read(c, typename traits::category<Coll>::type());
    }

  protected:
    template <typename Matrix> self& read(Matrix& A, tag::matrix);

    void set_symmetry(std::string& symmetry_text)
    {
	boost::to_lower(symmetry_text); 
	const char* symmetry_options[]= {"general", "symmetric", "skew-symmetric", "hermitian"};
	my_symmetry= string_to_enum(symmetry_text, symmetry_options, symmetry());
    }

    void set_sparsity(std::string& sparsity_text)
    {
	boost::to_lower(sparsity_text); 
	const char* sparsity_options[]= {"coordinate", "array"};
	my_sparsity= string_to_enum(sparsity_text, sparsity_options, sparsity());
    }

    template <typename Inserter, typename Value>
    void read_matrix(Inserter& ins, Value)
    {
	if (my_sparsity == coordinate) // sparse
	    while (my_stream) {
		int r, c;
		my_stream >> r >> c;
		insert_value(ins, r-1, c-1, Value());
	    }
	else // dense
	    for (int r= 0; r < nrows; r++)
		for (int c= 0; c < ncols; c++) 
		    insert_value(ins, r, c, Value());
    }

    template <typename Inserter, typename Value>
    void insert_value(Inserter& ins, int r, int c, Value) 
    {
	typedef typename Collection<typename Inserter::matrix_type>::value_type mvt;
	Value v;
	read_value(v);
	ins[r][c] << which_value(v, mvt());
	if (r != c) 
	    switch(my_symmetry) {
	      case symmetric:      ins[c][r] << which_value(v, mvt()); break;
	      case skew:           ins[c][r] << -which_value(v, mvt()); break;
	      case Hermitian:      ins[c][r] << conj(which_value(v, mvt())); break;
	    }
    }

    void read_value(pattern_type) {}
    void read_value(double& v) { my_stream >> v;}
    void read_value(long& v) { my_stream >> v;}
    void read_value(std::complex<double>& v) 
    { 
	double r, i; my_stream >> r >> i; v= std::complex<double>(r, i);
    }

    // Which value to be inserted? Itself if exist and 0 for pattern; complex are 
    template <typename Value, typename MValue> MValue which_value(Value v, MValue) { return v; }
    template <typename MValue> MValue which_value(pattern_type, MValue) { return 0.0; }
    template <typename MValue> MValue which_value(std::complex<double> v, MValue) { using std::abs; return abs(v); }
    std::complex<long double> which_value(std::complex<double> v, std::complex<long double>) { return v; }
    std::complex<double> which_value(std::complex<double> v, std::complex<double>) { return v; }
    std::complex<float> which_value(std::complex<double> v, std::complex<float>) { return std::complex<float>(real(v), imag(v)); }

    std::ifstream      my_stream;
    enum symmetry {general, symmetric, skew, Hermitian} my_symmetry;
    enum sparsity {coordinate, array} my_sparsity;
    std::size_t nrows, ncols, nnz;
};



// Matrix version
template <typename Matrix>
matrix_market_ifstream& matrix_market_ifstream::read(Matrix& A, tag::matrix)
{
    std::string marker, type, sparsity_text, value_format, symmetry_text;
    my_stream >> marker >> type >> sparsity_text >> value_format >> symmetry_text;
#if 0    
    std::cout << marker << ", " << type << ", " << sparsity_text << ", " 
	      << value_format << ", " << symmetry_text << "\n";
#endif
    MTL_THROW_IF(marker != std::string("%%MatrixMarket"), 
		 runtime_error("File not in Matrix Market format"));
    MTL_THROW_IF(type != std::string("matrix"), 
		 runtime_error("Try to read matrix from non-matrix file"));

    set_symmetry(symmetry_text);
    set_sparsity(sparsity_text);

    char first, comment[80];
    do {
	my_stream >> first;
	if (first == '%') // comments start with % -> ignore them
	    my_stream.getline(comment, 80, '\n'); // read rest of line
	else
	    my_stream.putback(first); // if not commment we still need it
    } while (first == '%');

    my_stream >> nrows >> ncols;
    if (sparsity_text == std::string("coordinate"))
	my_stream >> nnz;

    //std::cout << nrows << "x" << ncols << ", " << nnz << " non-zeros\n";	
    A.change_dim(nrows, ncols);
    set_to_zero(A);

    // Create enough space in sparse matrices; assumes row-major or square
    matrix::inserter<Matrix> ins(A, int(double(nnz) / double(nrows) * 1.3));

    if (value_format == std::string("real"))
	read_matrix(ins, double());
    else if (value_format == std::string("integer"))
	read_matrix(ins, long());
    else if (value_format == std::string("complex"))
	read_matrix(ins, std::complex<double>());
    else if (value_format == std::string("pattern"))
	read_matrix(ins, pattern_type());
    else
	MTL_THROW(runtime_error("Unknown tag for matrix value type in file"));

    return *this;
}

// To be implemented
class matrix_market_ofstream {};


// Have to go into a separate file
template <typename MatrixIFStream, typename MatrixOFStream>
class matrix_file
{
  public:
    explicit matrix_file(std::string fname) : fname(fname) {}
    explicit matrix_file(const char* fname) : fname(fname) {}

    std::string file_name() const { return fname; }

  protected:
    std::string fname;
};

// typedef matrix_file<matrix_market_ifstream, matrix_market_ofstream> matrix_market_file;



}} // namespace mtl::io

#endif // MTL_IO_MATRIX_MARKET_INCLUDE
