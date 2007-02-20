// $COPYRIGHT$

#ifndef MTL_SINGLE_SELECTOR_INCLUDE
#define MTL_SINGLE_SELECTOR_INCLUDE

namespace mtl {


template <typename Operation, typename Selection, typename Parameters, typename Backup>
struct single_selector
{
  // needs type !!!
};


template <typename Parameters, typename Backup>
struct single_selector<optag::dense_matmat_mult, hand_unrolled_22, Parameters, Backup>
{
    typedef gen_dense_matmat_mult_hand_22_t<Parameters::at<0>::type, .... // Matrices A, B, C
					    Parameters::at<3>::type,      // Assign mode
					    Backup>   type;
};


template <typename Parameters, typename Backup>
struct single_selector<optag::dense_matmat_mult, backend::blas, Parameters, Backup>
{
    typedef blas_dense_matmat_mult_t<Parameters::at<0>::type, .... // Matrices A, B, C
				     Parameters::at<3>::type,      // Assign mode
				     Backup>   type;
};


template <typename Parameters, typename Backup>
struct default_single_selector<optag::dense_matmat_mult, Parameters>
{
    typedef simple_dense_matmat_mult_t<Parameters::at<0>::type, .... // Matrices A, B, C
				       Parameters::at<3>::type      // Assign mode
    >   type;
};


} // namespace mtl

#endif // MTL_SINGLE_SELECTOR_INCLUDE
