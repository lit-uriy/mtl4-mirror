// Copyright Jeremy Siek  2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_DETAIL_UNION_JGS2006823_HPP
# define BOOST_DETAIL_UNION_JGS2006823_HPP

namespace boost { namespace detail {

  /*
    This algorithm is an n-way transform, except that it operates
    on indexed storage.
   */
  template<typename FSeqOfIndexed, typename IndexStream, typename BinaryFunction>
  void union_transform_all_sparse(FSeqOfIndexed sparse_inputs, IndexStream output, BinaryFunction f) {

    // while not at the end of all the ranges

    //   Find the next smallest index

    //   perform a fold over the values associated with the smallest index
    //   and increment all of those cursors

    //   output the index and accumulated value
    
  }

  template<typename FSeqOfIndexed, typename IndexStream, typename BinaryFunction>
  void intersect_transform_some_sparse(Indexed sparsest, FSeqOfIndexed other_inputs, IndexStream output,
				       BinaryFunction f) {

    // while not at the end of the sparsest

    //   loop, incrementing the other inputs to match the current index of sparsest

    //   perform a fold over the values associated with the current index

    //   output the index and accumulated value
    //   increment the cursor for sparsest 
    
  }


  template<typename FSeqOfIndexed, typename IndexStream, typename BinaryFunction>
  void transform_all_dense(FSeqOfIndexed dense_inputs, IndexStream output, BinaryFunction f) {

    // while not at the end 

    //   perform a fold over the values associated with the current cursors
    //   and increment all of the cursors

    //   output the index and accumulated value
  }
  
  
}} // namespace boost::detail

#endif // BOOST_DETAIL_UNION_JGS2006823_HPP
