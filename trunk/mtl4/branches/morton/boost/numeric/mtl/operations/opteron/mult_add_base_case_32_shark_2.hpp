
// $COPYRIGHT$

// Written by Michael Adams
// Modified by Peter Gottschling

#ifndef MTL_MULT_ADD_BASE_CASE_32_SHARK_2_INCLUDE
#define MTL_MULT_ADD_BASE_CASE_32_SHARK_2_INCLUDE

namespace mtl {


// #include <emmintrin.h>

// #include "schurBase-intr-shark.h"


struct mult_add_base_case_32_shark_2_opteron
{

  template <typename MatrixA, typename MatrixB, typename MatrixC>
  void operator()(MatrixA const& a, MatrixB const& b, MatrixC& c) 
  {
    // cast away const of a and b
    // ap= &a[0][0];

    // schurBase(cp, ap, bp);

    // if not divisible ?????????????
  }

private:
  void schurBase(double * D, double * C, double * BT)
  {
    const int baseOrder= 32,
              stride = baseOrder; 
    /*
    double * restrict D  =  aa + ((d*baseSize)&(rowMask|colMask));
    double * restrict C  =  aa + ((c*baseSize)&(rowMask|colMask));
    double * restrict BT =  aa + ((d*baseSize)&colMask)/2
      + ((c*baseSize)&colMask);
    */

  #if 0
    for (int i = 0; i < baseOrder; i+=2)
      for (int j = 0; j < baseOrder; j+=2)
        for (int k = 0; k < baseOrder; k++)
        {
  	D[0+(i)*stride+2*(j+0)] -= C[0+(i)*stride+2*k] * BT[0+(j)*stride+2*k];
  	D[0+(i)*stride+2*(j+1)] -= C[0+(i)*stride+2*k] * BT[1+(j)*stride+2*k];
  	D[1+(i)*stride+2*(j+0)] -= C[1+(i)*stride+2*k] * BT[0+(j)*stride+2*k];
  	D[1+(i)*stride+2*(j+1)] -= C[1+(i)*stride+2*k] * BT[1+(j)*stride+2*k];
        }
  #endif

  #if 0
    // Reorder loops (Ordering based on where the target code is going).
    for (int j = 0; j < baseOrder; j+=2)
      for (int i = 0; i < baseOrder; i+=16)
      {
        for (int k = 0; k < baseOrder; k++)
        {
          for (int i2 = i; i2 < i+16; i2+=2)
  	{
            D[0+(i2)*stride+2*(j+0)] -= C[0+(i2)*stride+2*k] * BT[0+(j)*stride+2*k];
            D[1+(i2)*stride+2*(j+0)] -= C[1+(i2)*stride+2*k] * BT[0+(j)*stride+2*k];
  	}
        }
        for (int k = 0; k < baseOrder; k++)
        {
          for (int i2 = i; i2 < i+16; i2+=2)
  	{
            D[0+(i2)*stride+2*(j+1)] -= C[0+(i2)*stride+2*k] * BT[1+(j)*stride+2*k];
            D[1+(i2)*stride+2*(j+1)] -= C[1+(i2)*stride+2*k] * BT[1+(j)*stride+2*k];
  	}
        }
      }
  #endif

  #if 0
    // Unroll i2

    for (int j = 0; j < baseOrder; j+=2)
      for (int i = 0; i < baseOrder; i+=16)
      {
        for (int k = 0; k < baseOrder; k++)
        {
          D[0+(i+ 0)*stride+2*(j+0)]-=C[0+(i+ 0)*stride+2*k]*BT[0+j*stride+2*k];
          D[1+(i+ 0)*stride+2*(j+0)]-=C[1+(i+ 0)*stride+2*k]*BT[0+j*stride+2*k];
          D[0+(i+ 2)*stride+2*(j+0)]-=C[0+(i+ 2)*stride+2*k]*BT[0+j*stride+2*k];
          D[1+(i+ 2)*stride+2*(j+0)]-=C[1+(i+ 2)*stride+2*k]*BT[0+j*stride+2*k];
          D[0+(i+ 4)*stride+2*(j+0)]-=C[0+(i+ 4)*stride+2*k]*BT[0+j*stride+2*k];
          D[1+(i+ 4)*stride+2*(j+0)]-=C[1+(i+ 4)*stride+2*k]*BT[0+j*stride+2*k];
          D[0+(i+ 6)*stride+2*(j+0)]-=C[0+(i+ 6)*stride+2*k]*BT[0+j*stride+2*k];
          D[1+(i+ 6)*stride+2*(j+0)]-=C[1+(i+ 6)*stride+2*k]*BT[0+j*stride+2*k];
          D[0+(i+ 8)*stride+2*(j+0)]-=C[0+(i+ 8)*stride+2*k]*BT[0+j*stride+2*k];
          D[1+(i+ 8)*stride+2*(j+0)]-=C[1+(i+ 8)*stride+2*k]*BT[0+j*stride+2*k];
          D[0+(i+10)*stride+2*(j+0)]-=C[0+(i+10)*stride+2*k]*BT[0+j*stride+2*k];
          D[1+(i+10)*stride+2*(j+0)]-=C[1+(i+10)*stride+2*k]*BT[0+j*stride+2*k];
          D[0+(i+12)*stride+2*(j+0)]-=C[0+(i+12)*stride+2*k]*BT[0+j*stride+2*k];
          D[1+(i+12)*stride+2*(j+0)]-=C[1+(i+12)*stride+2*k]*BT[0+j*stride+2*k];
          D[0+(i+14)*stride+2*(j+0)]-=C[0+(i+14)*stride+2*k]*BT[0+j*stride+2*k];
          D[1+(i+14)*stride+2*(j+0)]-=C[1+(i+14)*stride+2*k]*BT[0+j*stride+2*k];
        }
        for (int k = 0; k < baseOrder; k++)
        {
          D[0+(i+ 0)*stride+2*(j+1)]-=C[0+(i+ 0)*stride+2*k]*BT[1+j*stride+2*k];
          D[1+(i+ 0)*stride+2*(j+1)]-=C[1+(i+ 0)*stride+2*k]*BT[1+j*stride+2*k];
          D[0+(i+ 2)*stride+2*(j+1)]-=C[0+(i+ 2)*stride+2*k]*BT[1+j*stride+2*k];
          D[1+(i+ 2)*stride+2*(j+1)]-=C[1+(i+ 2)*stride+2*k]*BT[1+j*stride+2*k];
          D[0+(i+ 4)*stride+2*(j+1)]-=C[0+(i+ 4)*stride+2*k]*BT[1+j*stride+2*k];
          D[1+(i+ 4)*stride+2*(j+1)]-=C[1+(i+ 4)*stride+2*k]*BT[1+j*stride+2*k];
          D[0+(i+ 6)*stride+2*(j+1)]-=C[0+(i+ 6)*stride+2*k]*BT[1+j*stride+2*k];
          D[1+(i+ 6)*stride+2*(j+1)]-=C[1+(i+ 6)*stride+2*k]*BT[1+j*stride+2*k];
          D[0+(i+ 8)*stride+2*(j+1)]-=C[0+(i+ 8)*stride+2*k]*BT[1+j*stride+2*k];
          D[1+(i+ 8)*stride+2*(j+1)]-=C[1+(i+ 8)*stride+2*k]*BT[1+j*stride+2*k];
          D[0+(i+10)*stride+2*(j+1)]-=C[0+(i+10)*stride+2*k]*BT[1+j*stride+2*k];
          D[1+(i+10)*stride+2*(j+1)]-=C[1+(i+10)*stride+2*k]*BT[1+j*stride+2*k];
          D[0+(i+12)*stride+2*(j+1)]-=C[0+(i+12)*stride+2*k]*BT[1+j*stride+2*k];
          D[1+(i+12)*stride+2*(j+1)]-=C[1+(i+12)*stride+2*k]*BT[1+j*stride+2*k];
          D[0+(i+14)*stride+2*(j+1)]-=C[0+(i+14)*stride+2*k]*BT[1+j*stride+2*k];
          D[1+(i+14)*stride+2*(j+1)]-=C[1+(i+14)*stride+2*k]*BT[1+j*stride+2*k];
        }
      }
  #endif

  #if 0
    // Prep k
    for (int j = 0; j < baseOrder; j+=2)
      for (int i = 0; i < baseOrder; i+=16)
      {
        {
          double d00 = D[0+(i+ 0)*stride+2*(j+0)];
          double d01 = D[1+(i+ 0)*stride+2*(j+0)];
          double d02 = D[0+(i+ 2)*stride+2*(j+0)];
          double d03 = D[1+(i+ 2)*stride+2*(j+0)];
          double d04 = D[0+(i+ 4)*stride+2*(j+0)];
          double d05 = D[1+(i+ 4)*stride+2*(j+0)];
          double d06 = D[0+(i+ 6)*stride+2*(j+0)];
          double d07 = D[1+(i+ 6)*stride+2*(j+0)];
          double d08 = D[0+(i+ 8)*stride+2*(j+0)];
          double d09 = D[1+(i+ 8)*stride+2*(j+0)];
          double d10 = D[0+(i+10)*stride+2*(j+0)];
          double d11 = D[1+(i+10)*stride+2*(j+0)];
          double d12 = D[0+(i+12)*stride+2*(j+0)];
          double d13 = D[1+(i+12)*stride+2*(j+0)];
          double d14 = D[0+(i+14)*stride+2*(j+0)];
          double d15 = D[1+(i+14)*stride+2*(j+0)];
        for (int k = 0; k < baseOrder; k++)
        {
          d00-=C[0+(i+ 0)*stride+2*k]*BT[0+j*stride+2*k];
          d01-=C[1+(i+ 0)*stride+2*k]*BT[0+j*stride+2*k];
          d02-=C[0+(i+ 2)*stride+2*k]*BT[0+j*stride+2*k];
          d03-=C[1+(i+ 2)*stride+2*k]*BT[0+j*stride+2*k];
          d04-=C[0+(i+ 4)*stride+2*k]*BT[0+j*stride+2*k];
          d05-=C[1+(i+ 4)*stride+2*k]*BT[0+j*stride+2*k];
          d06-=C[0+(i+ 6)*stride+2*k]*BT[0+j*stride+2*k];
          d07-=C[1+(i+ 6)*stride+2*k]*BT[0+j*stride+2*k];
          d08-=C[0+(i+ 8)*stride+2*k]*BT[0+j*stride+2*k];
          d09-=C[1+(i+ 8)*stride+2*k]*BT[0+j*stride+2*k];
          d10-=C[0+(i+10)*stride+2*k]*BT[0+j*stride+2*k];
          d11-=C[1+(i+10)*stride+2*k]*BT[0+j*stride+2*k];
          d12-=C[0+(i+12)*stride+2*k]*BT[0+j*stride+2*k];
          d13-=C[1+(i+12)*stride+2*k]*BT[0+j*stride+2*k];
          d14-=C[0+(i+14)*stride+2*k]*BT[0+j*stride+2*k];
          d15-=C[1+(i+14)*stride+2*k]*BT[0+j*stride+2*k];
        }
          D[0+(i+ 0)*stride+2*(j+0)] = d00;
          D[1+(i+ 0)*stride+2*(j+0)] = d01;
          D[0+(i+ 2)*stride+2*(j+0)] = d02;
          D[1+(i+ 2)*stride+2*(j+0)] = d03;
          D[0+(i+ 4)*stride+2*(j+0)] = d04;
          D[1+(i+ 4)*stride+2*(j+0)] = d05;
          D[0+(i+ 6)*stride+2*(j+0)] = d06;
          D[1+(i+ 6)*stride+2*(j+0)] = d07;
          D[0+(i+ 8)*stride+2*(j+0)] = d08;
          D[1+(i+ 8)*stride+2*(j+0)] = d09;
          D[0+(i+10)*stride+2*(j+0)] = d10;
          D[1+(i+10)*stride+2*(j+0)] = d11;
          D[0+(i+12)*stride+2*(j+0)] = d12;
          D[1+(i+12)*stride+2*(j+0)] = d13;
          D[0+(i+14)*stride+2*(j+0)] = d14;
          D[1+(i+14)*stride+2*(j+0)] = d15;
        }

        {
          double d00 = D[0+(i+ 0)*stride+2*(j+1)];
          double d01 = D[1+(i+ 0)*stride+2*(j+1)];
          double d02 = D[0+(i+ 2)*stride+2*(j+1)];
          double d03 = D[1+(i+ 2)*stride+2*(j+1)];
          double d04 = D[0+(i+ 4)*stride+2*(j+1)];
          double d05 = D[1+(i+ 4)*stride+2*(j+1)];
          double d06 = D[0+(i+ 6)*stride+2*(j+1)];
          double d07 = D[1+(i+ 6)*stride+2*(j+1)];
          double d08 = D[0+(i+ 8)*stride+2*(j+1)];
          double d09 = D[1+(i+ 8)*stride+2*(j+1)];
          double d10 = D[0+(i+10)*stride+2*(j+1)];
          double d11 = D[1+(i+10)*stride+2*(j+1)];
          double d12 = D[0+(i+12)*stride+2*(j+1)];
          double d13 = D[1+(i+12)*stride+2*(j+1)];
          double d14 = D[0+(i+14)*stride+2*(j+1)];
          double d15 = D[1+(i+14)*stride+2*(j+1)];
        for (int k = 0; k < baseOrder; k++)
        {
          d00-=C[0+(i+ 0)*stride+2*k]*BT[1+j*stride+2*k];
          d01-=C[1+(i+ 0)*stride+2*k]*BT[1+j*stride+2*k];
          d02-=C[0+(i+ 2)*stride+2*k]*BT[1+j*stride+2*k];
          d03-=C[1+(i+ 2)*stride+2*k]*BT[1+j*stride+2*k];
          d04-=C[0+(i+ 4)*stride+2*k]*BT[1+j*stride+2*k];
          d05-=C[1+(i+ 4)*stride+2*k]*BT[1+j*stride+2*k];
          d06-=C[0+(i+ 6)*stride+2*k]*BT[1+j*stride+2*k];
          d07-=C[1+(i+ 6)*stride+2*k]*BT[1+j*stride+2*k];
          d08-=C[0+(i+ 8)*stride+2*k]*BT[1+j*stride+2*k];
          d09-=C[1+(i+ 8)*stride+2*k]*BT[1+j*stride+2*k];
          d10-=C[0+(i+10)*stride+2*k]*BT[1+j*stride+2*k];
          d11-=C[1+(i+10)*stride+2*k]*BT[1+j*stride+2*k];
          d12-=C[0+(i+12)*stride+2*k]*BT[1+j*stride+2*k];
          d13-=C[1+(i+12)*stride+2*k]*BT[1+j*stride+2*k];
          d14-=C[0+(i+14)*stride+2*k]*BT[1+j*stride+2*k];
          d15-=C[1+(i+14)*stride+2*k]*BT[1+j*stride+2*k];
        }
          D[0+(i+ 0)*stride+2*(j+1)] = d00;
          D[1+(i+ 0)*stride+2*(j+1)] = d01;
          D[0+(i+ 2)*stride+2*(j+1)] = d02;
          D[1+(i+ 2)*stride+2*(j+1)] = d03;
          D[0+(i+ 4)*stride+2*(j+1)] = d04;
          D[1+(i+ 4)*stride+2*(j+1)] = d05;
          D[0+(i+ 6)*stride+2*(j+1)] = d06;
          D[1+(i+ 6)*stride+2*(j+1)] = d07;
          D[0+(i+ 8)*stride+2*(j+1)] = d08;
          D[1+(i+ 8)*stride+2*(j+1)] = d09;
          D[0+(i+10)*stride+2*(j+1)] = d10;
          D[1+(i+10)*stride+2*(j+1)] = d11;
          D[0+(i+12)*stride+2*(j+1)] = d12;
          D[1+(i+12)*stride+2*(j+1)] = d13;
          D[0+(i+14)*stride+2*(j+1)] = d14;
          D[1+(i+14)*stride+2*(j+1)] = d15;
        }
      }
  #endif

  #if 0
    // Begin SSE
    for (int j = 0; j < baseOrder; j+=2)
      for (int i = 0; i < baseOrder; i+=16)
      {
        {
          __m128d d00 = _mm_load_pd(&D[0+(i+ 0)*stride+2*(j+0)]);
          __m128d d02 = _mm_load_pd(&D[0+(i+ 2)*stride+2*(j+0)]);
          __m128d d04 = _mm_load_pd(&D[0+(i+ 4)*stride+2*(j+0)]);
          __m128d d06 = _mm_load_pd(&D[0+(i+ 6)*stride+2*(j+0)]);
          __m128d d08 = _mm_load_pd(&D[0+(i+ 8)*stride+2*(j+0)]);
          __m128d d10 = _mm_load_pd(&D[0+(i+10)*stride+2*(j+0)]);
          __m128d d12 = _mm_load_pd(&D[0+(i+12)*stride+2*(j+0)]);
          __m128d d14 = _mm_load_pd(&D[0+(i+14)*stride+2*(j+0)]);
        for (int k = 0; k < baseOrder; k++)
        {
          __m128d bt0 = _mm_load1_pd(&BT[0+j*stride+2*k]);
  	d00-=_mm_load_pd(&C[0+(i+ 0)*stride+2*k])*bt0;
          d02-=_mm_load_pd(&C[0+(i+ 2)*stride+2*k])*bt0;
  	d04-=_mm_load_pd(&C[0+(i+ 4)*stride+2*k])*bt0;
  	d06-=_mm_load_pd(&C[0+(i+ 6)*stride+2*k])*bt0;
  	d08-=_mm_load_pd(&C[0+(i+ 8)*stride+2*k])*bt0;
  	d10-=_mm_load_pd(&C[0+(i+10)*stride+2*k])*bt0;
  	d12-=_mm_load_pd(&C[0+(i+12)*stride+2*k])*bt0;
  	d14-=_mm_load_pd(&C[0+(i+14)*stride+2*k])*bt0;
        }
          _mm_store_pd(&D[0+(i+ 0)*stride+2*(j+0)], d00);
          _mm_store_pd(&D[0+(i+ 2)*stride+2*(j+0)], d02);
          _mm_store_pd(&D[0+(i+ 4)*stride+2*(j+0)], d04);
          _mm_store_pd(&D[0+(i+ 6)*stride+2*(j+0)], d06);
          _mm_store_pd(&D[0+(i+ 8)*stride+2*(j+0)], d08);
          _mm_store_pd(&D[0+(i+10)*stride+2*(j+0)], d10);
          _mm_store_pd(&D[0+(i+12)*stride+2*(j+0)], d12);
          _mm_store_pd(&D[0+(i+14)*stride+2*(j+0)], d14);
        }

        {
          __m128d d00 = _mm_load_pd(&D[0+(i+ 0)*stride+2*(j+1)]);
          __m128d d02 = _mm_load_pd(&D[0+(i+ 2)*stride+2*(j+1)]);
          __m128d d04 = _mm_load_pd(&D[0+(i+ 4)*stride+2*(j+1)]);
          __m128d d06 = _mm_load_pd(&D[0+(i+ 6)*stride+2*(j+1)]);
          __m128d d08 = _mm_load_pd(&D[0+(i+ 8)*stride+2*(j+1)]);
          __m128d d10 = _mm_load_pd(&D[0+(i+10)*stride+2*(j+1)]);
          __m128d d12 = _mm_load_pd(&D[0+(i+12)*stride+2*(j+1)]);
          __m128d d14 = _mm_load_pd(&D[0+(i+14)*stride+2*(j+1)]);
        for (int k = 0; k < baseOrder; k++)
        {
          __m128d bt0 = _mm_load1_pd(&BT[1+j*stride+2*k]);
  	d00-=_mm_load_pd(&C[0+(i+ 0)*stride+2*k])*bt0;
          d02-=_mm_load_pd(&C[0+(i+ 2)*stride+2*k])*bt0;
  	d04-=_mm_load_pd(&C[0+(i+ 4)*stride+2*k])*bt0;
  	d06-=_mm_load_pd(&C[0+(i+ 6)*stride+2*k])*bt0;
  	d08-=_mm_load_pd(&C[0+(i+ 8)*stride+2*k])*bt0;
  	d10-=_mm_load_pd(&C[0+(i+10)*stride+2*k])*bt0;
  	d12-=_mm_load_pd(&C[0+(i+12)*stride+2*k])*bt0;
  	d14-=_mm_load_pd(&C[0+(i+14)*stride+2*k])*bt0;
        }
          _mm_store_pd(&D[0+(i+ 0)*stride+2*(j+1)], d00);
          _mm_store_pd(&D[0+(i+ 2)*stride+2*(j+1)], d02);
          _mm_store_pd(&D[0+(i+ 4)*stride+2*(j+1)], d04);
          _mm_store_pd(&D[0+(i+ 6)*stride+2*(j+1)], d06);
          _mm_store_pd(&D[0+(i+ 8)*stride+2*(j+1)], d08);
          _mm_store_pd(&D[0+(i+10)*stride+2*(j+1)], d10);
          _mm_store_pd(&D[0+(i+12)*stride+2*(j+1)], d12);
          _mm_store_pd(&D[0+(i+14)*stride+2*(j+1)], d14);
        }
      }
  #endif

  #if 0
    // Tweak SSE
  #define MM_LOAD1_PD(a,b) \
  { \
    __asm__("movlpd %1, %0" : "=x" (a) : "m"(*b)); \
    __asm__("movhpd %1, %0" : "=x" (a) : "m"(*b), "0" (a)); \
  }
  #define MM_LOAD1U_PD(a,b) \
  { \
    __asm__("movlpd %1, %0" : "=x" (a) : "m"(*b)); \
    __asm__("movhpd %1, %0" : "=x" (a) : "m"(*b), "0" (a)); \
  }
  #define MM_MUL_PD(out,addr) \
  { out = _mm_mul_pd(out, *(__m128d*)addr); }
    for (int j = 0; j < baseOrder; j+=2)
      for (int i = 0; i < baseOrder; i+=16)
      {
        {
          __m128d d00 = _mm_load_pd(&D[0+(i+ 0)*stride+2*(j+0)]);
          __m128d d02 = _mm_load_pd(&D[0+(i+ 2)*stride+2*(j+0)]);
          __m128d d04 = _mm_load_pd(&D[0+(i+ 4)*stride+2*(j+0)]);
          __m128d d06 = _mm_load_pd(&D[0+(i+ 6)*stride+2*(j+0)]);
          __m128d d08 = _mm_load_pd(&D[0+(i+ 8)*stride+2*(j+0)]);
          __m128d d10 = _mm_load_pd(&D[0+(i+10)*stride+2*(j+0)]);
          __m128d d12 = _mm_load_pd(&D[0+(i+12)*stride+2*(j+0)]);
          __m128d d14 = _mm_load_pd(&D[0+(i+14)*stride+2*(j+0)]);
        for (int k = 0; k < baseOrder; k++)
        {
          __m128d bt0;
          MM_LOAD1_PD(bt0, &BT[0+j*stride+2*k]);
  	d00-=_mm_load_pd(&C[0+(i+ 0)*stride+2*k])*bt0;
          d02-=_mm_load_pd(&C[0+(i+ 2)*stride+2*k])*bt0;
  	d04-=_mm_load_pd(&C[0+(i+ 4)*stride+2*k])*bt0;
  	d06-=_mm_load_pd(&C[0+(i+ 6)*stride+2*k])*bt0;
  	d08-=_mm_load_pd(&C[0+(i+ 8)*stride+2*k])*bt0;
  	d10-=_mm_load_pd(&C[0+(i+10)*stride+2*k])*bt0;
  	d12-=_mm_load_pd(&C[0+(i+12)*stride+2*k])*bt0;
          MM_MUL_PD(bt0, &C[0+(i+14)*stride+2*k]);
          d14-=bt0;
        }
          _mm_store_pd(&D[0+(i+ 0)*stride+2*(j+0)], d00);
          _mm_store_pd(&D[0+(i+ 2)*stride+2*(j+0)], d02);
          _mm_store_pd(&D[0+(i+ 4)*stride+2*(j+0)], d04);
          _mm_store_pd(&D[0+(i+ 6)*stride+2*(j+0)], d06);
          _mm_store_pd(&D[0+(i+ 8)*stride+2*(j+0)], d08);
          _mm_store_pd(&D[0+(i+10)*stride+2*(j+0)], d10);
          _mm_store_pd(&D[0+(i+12)*stride+2*(j+0)], d12);
          _mm_store_pd(&D[0+(i+14)*stride+2*(j+0)], d14);
        }

        {
          __m128d d00 = _mm_load_pd(&D[0+(i+ 0)*stride+2*(j+1)]);
          __m128d d02 = _mm_load_pd(&D[0+(i+ 2)*stride+2*(j+1)]);
          __m128d d04 = _mm_load_pd(&D[0+(i+ 4)*stride+2*(j+1)]);
          __m128d d06 = _mm_load_pd(&D[0+(i+ 6)*stride+2*(j+1)]);
          __m128d d08 = _mm_load_pd(&D[0+(i+ 8)*stride+2*(j+1)]);
          __m128d d10 = _mm_load_pd(&D[0+(i+10)*stride+2*(j+1)]);
          __m128d d12 = _mm_load_pd(&D[0+(i+12)*stride+2*(j+1)]);
          __m128d d14 = _mm_load_pd(&D[0+(i+14)*stride+2*(j+1)]);
        for (int k = 0; k < baseOrder; k++)
        {
          __m128d bt1;
  	MM_LOAD1U_PD(bt1, &BT[1+j*stride+2*k]);
  	d00-=_mm_load_pd(&C[0+(i+ 0)*stride+2*k])*bt1;
          d02-=_mm_load_pd(&C[0+(i+ 2)*stride+2*k])*bt1;
  	d04-=_mm_load_pd(&C[0+(i+ 4)*stride+2*k])*bt1;
  	d06-=_mm_load_pd(&C[0+(i+ 6)*stride+2*k])*bt1;
  	d08-=_mm_load_pd(&C[0+(i+ 8)*stride+2*k])*bt1;
  	d10-=_mm_load_pd(&C[0+(i+10)*stride+2*k])*bt1;
  	d12-=_mm_load_pd(&C[0+(i+12)*stride+2*k])*bt1;
          MM_MUL_PD(bt1, &C[0+(i+14)*stride+2*k]);
  	d14-=bt1;
        }
          _mm_store_pd(&D[0+(i+ 0)*stride+2*(j+1)], d00);
          _mm_store_pd(&D[0+(i+ 2)*stride+2*(j+1)], d02);
          _mm_store_pd(&D[0+(i+ 4)*stride+2*(j+1)], d04);
          _mm_store_pd(&D[0+(i+ 6)*stride+2*(j+1)], d06);
          _mm_store_pd(&D[0+(i+ 8)*stride+2*(j+1)], d08);
          _mm_store_pd(&D[0+(i+10)*stride+2*(j+1)], d10);
          _mm_store_pd(&D[0+(i+12)*stride+2*(j+1)], d12);
          _mm_store_pd(&D[0+(i+14)*stride+2*(j+1)], d14);
        }
      }
  #endif

  #if 0
    // Factor and unroll k
  #define MM_LOAD1_PD(a,b) \
  { \
    __asm__("movlpd %1, %0" : "=x" (a) : "m"(*b)); \
    __asm__("movhpd %1, %0" : "=x" (a) : "m"(*b), "0" (a)); \
  }
  #define MM_LOAD1U_PD(a,b) \
  { \
    __asm__("movlpd %1, %0" : "=x" (a) : "m"(*b)); \
    __asm__("movhpd %1, %0" : "=x" (a) : "m"(*b), "0" (a)); \
  }
  #define MM_MUL_PD(out,addr) \
  { out = _mm_mul_pd(out, *(__m128d*)addr); }
  #define BLOCK0_0(i,j,k) \
        { \
          __m128d bt0; \
          MM_LOAD1_PD(bt0, &BT[0+j*stride+2*k]); \
  	d00-=_mm_load_pd(&C[0+(i+ 0)*stride+2*k])*bt0; \
          d02-=_mm_load_pd(&C[0+(i+ 2)*stride+2*k])*bt0; \
  	d04-=_mm_load_pd(&C[0+(i+ 4)*stride+2*k])*bt0; \
  	d06-=_mm_load_pd(&C[0+(i+ 6)*stride+2*k])*bt0; \
  	d08-=_mm_load_pd(&C[0+(i+ 8)*stride+2*k])*bt0; \
  	d10-=_mm_load_pd(&C[0+(i+10)*stride+2*k])*bt0; \
  	d12-=_mm_load_pd(&C[0+(i+12)*stride+2*k])*bt0; \
          MM_MUL_PD(bt0, &C[0+(i+14)*stride+2*k]); \
          d14-=bt0; \
        }
  #define BLOCK0_1(i,j,k) \
        { \
          __m128d bt1; \
  	MM_LOAD1U_PD(bt1, &BT[1+j*stride+2*k]); \
  	d00-=_mm_load_pd(&C[0+(i+ 0)*stride+2*k])*bt1; \
          d02-=_mm_load_pd(&C[0+(i+ 2)*stride+2*k])*bt1; \
  	d04-=_mm_load_pd(&C[0+(i+ 4)*stride+2*k])*bt1; \
  	d06-=_mm_load_pd(&C[0+(i+ 6)*stride+2*k])*bt1; \
  	d08-=_mm_load_pd(&C[0+(i+ 8)*stride+2*k])*bt1; \
  	d10-=_mm_load_pd(&C[0+(i+10)*stride+2*k])*bt1; \
  	d12-=_mm_load_pd(&C[0+(i+12)*stride+2*k])*bt1; \
          MM_MUL_PD(bt1, &C[0+(i+14)*stride+2*k]); \
  	d14-=bt1; \
        }
    for (int j = 0; j < baseOrder; j+=2)
      for (int i = 0; i < baseOrder; i+=16)
      {
        {
          __m128d d00 = _mm_load_pd(&D[0+(i+ 0)*stride+2*(j+0)]);
          __m128d d02 = _mm_load_pd(&D[0+(i+ 2)*stride+2*(j+0)]);
          __m128d d04 = _mm_load_pd(&D[0+(i+ 4)*stride+2*(j+0)]);
          __m128d d06 = _mm_load_pd(&D[0+(i+ 6)*stride+2*(j+0)]);
          __m128d d08 = _mm_load_pd(&D[0+(i+ 8)*stride+2*(j+0)]);
          __m128d d10 = _mm_load_pd(&D[0+(i+10)*stride+2*(j+0)]);
          __m128d d12 = _mm_load_pd(&D[0+(i+12)*stride+2*(j+0)]);
          __m128d d14 = _mm_load_pd(&D[0+(i+14)*stride+2*(j+0)]);
  	for (int k = 0; k < baseOrder; k+=32)
  	{
  	  BLOCK0_0(i,j,(k+ 0));
  	  BLOCK0_0(i,j,(k+ 1));
  	  BLOCK0_0(i,j,(k+ 2));
  	  BLOCK0_0(i,j,(k+ 3));
  	  BLOCK0_0(i,j,(k+ 4));
  	  BLOCK0_0(i,j,(k+ 5));
  	  BLOCK0_0(i,j,(k+ 6));
  	  BLOCK0_0(i,j,(k+ 7));
  	  BLOCK0_0(i,j,(k+ 8));
  	  BLOCK0_0(i,j,(k+ 9));
  	  BLOCK0_0(i,j,(k+10));
  	  BLOCK0_0(i,j,(k+11));
  	  BLOCK0_0(i,j,(k+12));
  	  BLOCK0_0(i,j,(k+13));
  	  BLOCK0_0(i,j,(k+14));
  	  BLOCK0_0(i,j,(k+15));
  	  BLOCK0_0(i,j,(k+16));
  	  BLOCK0_0(i,j,(k+17));
  	  BLOCK0_0(i,j,(k+18));
  	  BLOCK0_0(i,j,(k+19));
  	  BLOCK0_0(i,j,(k+20));
  	  BLOCK0_0(i,j,(k+21));
  	  BLOCK0_0(i,j,(k+22));
  	  BLOCK0_0(i,j,(k+23));
  	  BLOCK0_0(i,j,(k+24));
  	  BLOCK0_0(i,j,(k+25));
  	  BLOCK0_0(i,j,(k+26));
  	  BLOCK0_0(i,j,(k+27));
  	  BLOCK0_0(i,j,(k+28));
  	  BLOCK0_0(i,j,(k+29));
  	  BLOCK0_0(i,j,(k+30));
  	  BLOCK0_0(i,j,(k+31));
  	}
          _mm_store_pd(&D[0+(i+ 0)*stride+2*(j+0)], d00);
          _mm_store_pd(&D[0+(i+ 2)*stride+2*(j+0)], d02);
          _mm_store_pd(&D[0+(i+ 4)*stride+2*(j+0)], d04);
          _mm_store_pd(&D[0+(i+ 6)*stride+2*(j+0)], d06);
          _mm_store_pd(&D[0+(i+ 8)*stride+2*(j+0)], d08);
          _mm_store_pd(&D[0+(i+10)*stride+2*(j+0)], d10);
          _mm_store_pd(&D[0+(i+12)*stride+2*(j+0)], d12);
          _mm_store_pd(&D[0+(i+14)*stride+2*(j+0)], d14);
        }

        {
          __m128d d00 = _mm_load_pd(&D[0+(i+ 0)*stride+2*(j+1)]);
          __m128d d02 = _mm_load_pd(&D[0+(i+ 2)*stride+2*(j+1)]);
          __m128d d04 = _mm_load_pd(&D[0+(i+ 4)*stride+2*(j+1)]);
          __m128d d06 = _mm_load_pd(&D[0+(i+ 6)*stride+2*(j+1)]);
          __m128d d08 = _mm_load_pd(&D[0+(i+ 8)*stride+2*(j+1)]);
          __m128d d10 = _mm_load_pd(&D[0+(i+10)*stride+2*(j+1)]);
          __m128d d12 = _mm_load_pd(&D[0+(i+12)*stride+2*(j+1)]);
          __m128d d14 = _mm_load_pd(&D[0+(i+14)*stride+2*(j+1)]);
  	for (int k = 0; k < baseOrder; k+=32)
  	{
  	  BLOCK0_1(i,j,(k+ 0));
  	  BLOCK0_1(i,j,(k+ 1));
  	  BLOCK0_1(i,j,(k+ 2));
  	  BLOCK0_1(i,j,(k+ 3));
  	  BLOCK0_1(i,j,(k+ 4));
  	  BLOCK0_1(i,j,(k+ 5));
  	  BLOCK0_1(i,j,(k+ 6));
  	  BLOCK0_1(i,j,(k+ 7));
  	  BLOCK0_1(i,j,(k+ 8));
  	  BLOCK0_1(i,j,(k+ 9));
  	  BLOCK0_1(i,j,(k+10));
  	  BLOCK0_1(i,j,(k+11));
  	  BLOCK0_1(i,j,(k+12));
  	  BLOCK0_1(i,j,(k+13));
  	  BLOCK0_1(i,j,(k+14));
  	  BLOCK0_1(i,j,(k+15));
  	  BLOCK0_1(i,j,(k+16));
  	  BLOCK0_1(i,j,(k+17));
  	  BLOCK0_1(i,j,(k+18));
  	  BLOCK0_1(i,j,(k+19));
  	  BLOCK0_1(i,j,(k+20));
  	  BLOCK0_1(i,j,(k+21));
  	  BLOCK0_1(i,j,(k+22));
  	  BLOCK0_1(i,j,(k+23));
  	  BLOCK0_1(i,j,(k+24));
  	  BLOCK0_1(i,j,(k+25));
  	  BLOCK0_1(i,j,(k+26));
  	  BLOCK0_1(i,j,(k+27));
  	  BLOCK0_1(i,j,(k+28));
  	  BLOCK0_1(i,j,(k+29));
  	  BLOCK0_1(i,j,(k+30));
  	  BLOCK0_1(i,j,(k+31));
  	}
          _mm_store_pd(&D[0+(i+ 0)*stride+2*(j+1)], d00);
          _mm_store_pd(&D[0+(i+ 2)*stride+2*(j+1)], d02);
          _mm_store_pd(&D[0+(i+ 4)*stride+2*(j+1)], d04);
          _mm_store_pd(&D[0+(i+ 6)*stride+2*(j+1)], d06);
          _mm_store_pd(&D[0+(i+ 8)*stride+2*(j+1)], d08);
          _mm_store_pd(&D[0+(i+10)*stride+2*(j+1)], d10);
          _mm_store_pd(&D[0+(i+12)*stride+2*(j+1)], d12);
          _mm_store_pd(&D[0+(i+14)*stride+2*(j+1)], d14);
        }
      }
  #endif

  #if 1
    // Factor and unroll i
  #define MM_LOAD1_PD(a,b) \
  { \
    __asm__("movlpd %1, %0" : "=x" (a) : "m"(*b)); \
    __asm__("movhpd %1, %0" : "=x" (a) : "m"(*b), "0" (a)); \
  }
  #define MM_LOAD1U_PD(a,b) \
  { \
    __asm__("movlpd %1, %0" : "=x" (a) : "m"(*b)); \
    __asm__("movhpd %1, %0" : "=x" (a) : "m"(*b), "0" (a)); \
  }
  #define MM_MUL_PD(out,addr) \
  { out = _mm_mul_pd(out, *(__m128d*)addr); }
  #define BLOCK0_0(i,j,k) \
        { \
          __m128d bt0; \
          MM_LOAD1_PD(bt0, &BT[0+j*stride+2*k]); \
  	d00-=_mm_load_pd(&C[0+(i+ 0)*stride+2*k])*bt0; \
          d02-=_mm_load_pd(&C[0+(i+ 2)*stride+2*k])*bt0; \
  	d04-=_mm_load_pd(&C[0+(i+ 4)*stride+2*k])*bt0; \
  	d06-=_mm_load_pd(&C[0+(i+ 6)*stride+2*k])*bt0; \
  	d08-=_mm_load_pd(&C[0+(i+ 8)*stride+2*k])*bt0; \
  	d10-=_mm_load_pd(&C[0+(i+10)*stride+2*k])*bt0; \
  	d12-=_mm_load_pd(&C[0+(i+12)*stride+2*k])*bt0; \
          MM_MUL_PD(bt0, &C[0+(i+14)*stride+2*k]); \
          d14-=bt0; \
        }
  #define BLOCK0_1(i,j,k) \
        { \
          __m128d bt1; \
  	MM_LOAD1U_PD(bt1, &BT[1+j*stride+2*k]); \
  	d00-=_mm_load_pd(&C[0+(i+ 0)*stride+2*k])*bt1; \
          d02-=_mm_load_pd(&C[0+(i+ 2)*stride+2*k])*bt1; \
  	d04-=_mm_load_pd(&C[0+(i+ 4)*stride+2*k])*bt1; \
  	d06-=_mm_load_pd(&C[0+(i+ 6)*stride+2*k])*bt1; \
  	d08-=_mm_load_pd(&C[0+(i+ 8)*stride+2*k])*bt1; \
  	d10-=_mm_load_pd(&C[0+(i+10)*stride+2*k])*bt1; \
  	d12-=_mm_load_pd(&C[0+(i+12)*stride+2*k])*bt1; \
          MM_MUL_PD(bt1, &C[0+(i+14)*stride+2*k]); \
  	d14-=bt1; \
        }
  #define BLOCK1_0(i,j) \
        { \
          __m128d d00 = _mm_load_pd(&D[0+(i+ 0)*stride+2*(j+0)]); \
          __m128d d02 = _mm_load_pd(&D[0+(i+ 2)*stride+2*(j+0)]); \
          __m128d d04 = _mm_load_pd(&D[0+(i+ 4)*stride+2*(j+0)]); \
          __m128d d06 = _mm_load_pd(&D[0+(i+ 6)*stride+2*(j+0)]); \
          __m128d d08 = _mm_load_pd(&D[0+(i+ 8)*stride+2*(j+0)]); \
          __m128d d10 = _mm_load_pd(&D[0+(i+10)*stride+2*(j+0)]); \
          __m128d d12 = _mm_load_pd(&D[0+(i+12)*stride+2*(j+0)]); \
          __m128d d14 = _mm_load_pd(&D[0+(i+14)*stride+2*(j+0)]); \
  	for (int k = 0; k < baseOrder; k+=32) \
  	{ \
  	  BLOCK0_0(i,j,(k+ 0)); \
  	  BLOCK0_0(i,j,(k+ 1)); \
  	  BLOCK0_0(i,j,(k+ 2)); \
  	  BLOCK0_0(i,j,(k+ 3)); \
  	  BLOCK0_0(i,j,(k+ 4)); \
  	  BLOCK0_0(i,j,(k+ 5)); \
  	  BLOCK0_0(i,j,(k+ 6)); \
  	  BLOCK0_0(i,j,(k+ 7)); \
  	  BLOCK0_0(i,j,(k+ 8)); \
  	  BLOCK0_0(i,j,(k+ 9)); \
  	  BLOCK0_0(i,j,(k+10)); \
  	  BLOCK0_0(i,j,(k+11)); \
  	  BLOCK0_0(i,j,(k+12)); \
  	  BLOCK0_0(i,j,(k+13)); \
  	  BLOCK0_0(i,j,(k+14)); \
  	  BLOCK0_0(i,j,(k+15)); \
  	  BLOCK0_0(i,j,(k+16)); \
  	  BLOCK0_0(i,j,(k+17)); \
  	  BLOCK0_0(i,j,(k+18)); \
  	  BLOCK0_0(i,j,(k+19)); \
  	  BLOCK0_0(i,j,(k+20)); \
  	  BLOCK0_0(i,j,(k+21)); \
  	  BLOCK0_0(i,j,(k+22)); \
  	  BLOCK0_0(i,j,(k+23)); \
  	  BLOCK0_0(i,j,(k+24)); \
  	  BLOCK0_0(i,j,(k+25)); \
  	  BLOCK0_0(i,j,(k+26)); \
  	  BLOCK0_0(i,j,(k+27)); \
  	  BLOCK0_0(i,j,(k+28)); \
  	  BLOCK0_0(i,j,(k+29)); \
  	  BLOCK0_0(i,j,(k+30)); \
  	  BLOCK0_0(i,j,(k+31)); \
  	} \
          _mm_store_pd(&D[0+(i+ 0)*stride+2*(j+0)], d00); \
          _mm_store_pd(&D[0+(i+ 2)*stride+2*(j+0)], d02); \
          _mm_store_pd(&D[0+(i+ 4)*stride+2*(j+0)], d04); \
          _mm_store_pd(&D[0+(i+ 6)*stride+2*(j+0)], d06); \
          _mm_store_pd(&D[0+(i+ 8)*stride+2*(j+0)], d08); \
          _mm_store_pd(&D[0+(i+10)*stride+2*(j+0)], d10); \
          _mm_store_pd(&D[0+(i+12)*stride+2*(j+0)], d12); \
          _mm_store_pd(&D[0+(i+14)*stride+2*(j+0)], d14); \
        }
  #define BLOCK1_1(i,j) \
        { \
          __m128d d00 = _mm_load_pd(&D[0+(i+ 0)*stride+2*(j+1)]); \
          __m128d d02 = _mm_load_pd(&D[0+(i+ 2)*stride+2*(j+1)]); \
          __m128d d04 = _mm_load_pd(&D[0+(i+ 4)*stride+2*(j+1)]); \
          __m128d d06 = _mm_load_pd(&D[0+(i+ 6)*stride+2*(j+1)]); \
          __m128d d08 = _mm_load_pd(&D[0+(i+ 8)*stride+2*(j+1)]); \
          __m128d d10 = _mm_load_pd(&D[0+(i+10)*stride+2*(j+1)]); \
          __m128d d12 = _mm_load_pd(&D[0+(i+12)*stride+2*(j+1)]); \
          __m128d d14 = _mm_load_pd(&D[0+(i+14)*stride+2*(j+1)]); \
  	for (int k = 0; k < baseOrder; k+=32) \
  	{ \
  	  BLOCK0_1(i,j,(k+ 0)); \
  	  BLOCK0_1(i,j,(k+ 1)); \
  	  BLOCK0_1(i,j,(k+ 2)); \
  	  BLOCK0_1(i,j,(k+ 3)); \
  	  BLOCK0_1(i,j,(k+ 4)); \
  	  BLOCK0_1(i,j,(k+ 5)); \
  	  BLOCK0_1(i,j,(k+ 6)); \
  	  BLOCK0_1(i,j,(k+ 7)); \
  	  BLOCK0_1(i,j,(k+ 8)); \
  	  BLOCK0_1(i,j,(k+ 9)); \
  	  BLOCK0_1(i,j,(k+10)); \
  	  BLOCK0_1(i,j,(k+11)); \
  	  BLOCK0_1(i,j,(k+12)); \
  	  BLOCK0_1(i,j,(k+13)); \
  	  BLOCK0_1(i,j,(k+14)); \
  	  BLOCK0_1(i,j,(k+15)); \
  	  BLOCK0_1(i,j,(k+16)); \
  	  BLOCK0_1(i,j,(k+17)); \
  	  BLOCK0_1(i,j,(k+18)); \
  	  BLOCK0_1(i,j,(k+19)); \
  	  BLOCK0_1(i,j,(k+20)); \
  	  BLOCK0_1(i,j,(k+21)); \
  	  BLOCK0_1(i,j,(k+22)); \
  	  BLOCK0_1(i,j,(k+23)); \
  	  BLOCK0_1(i,j,(k+24)); \
  	  BLOCK0_1(i,j,(k+25)); \
  	  BLOCK0_1(i,j,(k+26)); \
  	  BLOCK0_1(i,j,(k+27)); \
  	  BLOCK0_1(i,j,(k+28)); \
  	  BLOCK0_1(i,j,(k+29)); \
  	  BLOCK0_1(i,j,(k+30)); \
  	  BLOCK0_1(i,j,(k+31)); \
  	} \
          _mm_store_pd(&D[0+(i+ 0)*stride+2*(j+1)], d00); \
          _mm_store_pd(&D[0+(i+ 2)*stride+2*(j+1)], d02); \
          _mm_store_pd(&D[0+(i+ 4)*stride+2*(j+1)], d04); \
          _mm_store_pd(&D[0+(i+ 6)*stride+2*(j+1)], d06); \
          _mm_store_pd(&D[0+(i+ 8)*stride+2*(j+1)], d08); \
          _mm_store_pd(&D[0+(i+10)*stride+2*(j+1)], d10); \
          _mm_store_pd(&D[0+(i+12)*stride+2*(j+1)], d12); \
          _mm_store_pd(&D[0+(i+14)*stride+2*(j+1)], d14); \
        }

    for (int j = 0; j < baseOrder; j+=2)
    {
      BLOCK1_0( 0,j);
      BLOCK1_1( 0,j);
      BLOCK1_0(16,j);
      BLOCK1_1(16,j);
    }
  #endif

  }
   
  
};

} // namespace mtl

#endif // MTL_MULT_ADD_BASE_CASE_32_SHARK_2_INCLUDE
