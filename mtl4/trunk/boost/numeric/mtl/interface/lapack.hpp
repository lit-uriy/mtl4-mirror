// $COPYRIGHT$

#ifndef MTL_LAPACK_INCLUDE
#define MTL_LAPACK_INCLUDE

#include <complex>


#ifdef __cplusplus
extern "C" {
#endif


  // Cholesky Factorization
  void spotrf_(const char* uplo, const int* n, float *a, const int* ld, int* info);
  void dpotrf_(const char* uplo, const int* n, double *a, const int* ld, int* info);
  void cpotrf_(const char* uplo, const int* n, std::complex<float> *a, const int* ld, int* info);
  void zpotrf_(const char* uplo, const int* n, std::complex<double> *a, const int* ld, int* info);














#ifdef __cplusplus
} // extern "C"
#endif


#endif // MTL_LAPACK_INCLUDE
