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

#ifndef MTL_IO_MATRIX_FILE_INCLUDE
#define MTL_IO_MATRIX_FILE_INCLUDE

#ifdef MTL_HAS_MPI
#   include <boost/numeric/mtl/par/distribution.hpp> 
#endif

namespace mtl { namespace io {

template <typename MatrixIFStream, typename MatrixOFStream>
class matrix_file
{
    typedef matrix_file self;
  public:
    explicit matrix_file(const std::string& fname) : fname(fname) {}
    explicit matrix_file(const char* fname) : fname(fname) {}

    std::string file_name() const { return fname; }

    template <typename Collection>
    matrix_file& operator=(const Collection& c)
    {
	MatrixOFStream stream(fname);
	stream << c;
	return *this;
    }

#ifdef MTL_HAS_MPI
    // Not really elagant, should be refactored some day
    friend inline std::size_t num_rows(const self&) { return 0; }
    friend inline std::size_t num_cols(const self&) { return 0; }
    friend inline par::block_distribution row_distribution(const self&) { return par::block_distribution(0); }
    friend inline par::block_distribution col_distribution(const self&) { return par::block_distribution(0); }
    friend inline bool referred_distribution(const self&) { return false; }
#endif

  protected:
    std::string fname;
};

}} // namespace mtl::io

#endif // MTL_IO_MATRIX_FILE_INCLUDE
