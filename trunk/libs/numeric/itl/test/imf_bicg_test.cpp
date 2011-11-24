// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.
//
// Algorithm inspired by Nick Vannieuwenhoven, written by Cornelius Steinhardt

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>
#include <boost/numeric/itl/pc/io.hpp>
#include <boost/numeric/itl/pc/matrix_algorithms.hpp>
#include <boost/numeric/itl/pc/imf_preconditioner.hpp>
#include <boost/numeric/itl/pc/imf_algorithms.hpp>

template< class ElementStructure >
void setup(ElementStructure& es, int lofi)
{
    typedef double value_type;
      
    int size( es.get_total_vars() );
    mtl::dense_vector<int> ident(size);
    iota(ident);
    mtl::compressed2D<value_type>* master_mat(mtl::matrix::assemble_compressed(es,ident)); 
    itl::pc::imf_preconditioner<value_type> precond(es, lofi);

#if 0
	mtl::io::tout << "------------------------------- STATISTICS -------------------------------" << std::endl;
	int rows = num_rows(*master_mat);
	int cols = num_cols(*master_mat);
 	int nnz = (*master_mat).nnz();
	mtl::io::tout << "Dimensions: " << rows << " x " << cols << std::endl;
	mtl::io::tout << "Non-zeros: " << nnz << std::endl;
	mtl::io::tout << "Sparsity (%): " << ((double(nnz) / rows) / cols) << std::endl;
	mtl::io::tout << "Avg nnz/row: " << double(nnz) / rows << std::endl;
	mtl::io::tout << std::endl;
	mtl::io::tout << "Elements: " << es.get_total_elements() << std::endl;
	mtl::io::tout << "Variables: " << es.get_total_vars() << std::endl;
	mtl::io::tout << "--------------------------------------------------------------------------" << std::endl;
// calculate eigenvalues
	mtl::dense2D<value_type> E(size,size),A(*master_mat);
	for(int i=0; i<size;i++){
	  mtl::dense_vector<value_type> tmp(A[mtl::irange(0, mtl::imax)][i]);
	  E[mtl::irange(0, mtl::imax)][i] = precond.solve(tmp);
	}
	mtl::io::tout<< "E=\n"<<E <<"\n";
#endif
	
    mtl::dense_vector<value_type>              x(size, 1), b(size);
    b= *master_mat * x;
    itl::cyclic_iteration<value_type>          iter(b, size, 1.e-8, 0.0, 5);
    x= 0;
    bicgstab(*master_mat, x, b, precond, iter);

}

int main(int, char** argv)
{
 
    typedef double value_type;
    typedef mtl::compressed2D<value_type>     sparse_type;
       
    std::string program_dir= mtl::io::directory_name(argv[0]),
	        matrix_file= mtl::io::join(program_dir, "../../mtl/test/matrix_market/square3.mtx");

    // matrix_file="/home/cornelius/projects/diplom/parallel_mtl4/libs/numeric/mtl/mpi_test/matrix_market/square3.mtx";

    mtl::element_structure<value_type>* es = 0;
//     mtl::io::tout<< "matrix_file=" << matrix_file.c_str()  << "\n";

    es = mtl::read_el_matrix<value_type>(matrix_file.c_str());
    int lofi=3;
	
    setup(*es, lofi);
    return 0;
}
