/*****************************************************************************
  file: utilities.h
  -----------------
  Includes all files, declares all variables, and defines all macros and
    functions necessary for the design and application of the sparse
    quadtree representation.
  The global variables declared as external are pre-defined in the
  "main"("executing") file' s header (here: sysSolver.h)

  Revised on: 07/25/06

  D. S. Wise
  Larisse D. Voufo
*****************************************************************************/
#ifndef UTILITIES_H
#define UTILITIES_H

//include files
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <cassert>

#include "defs.h"
#include "timestamp.h"
#include "dilate.h"
#include "memory.h"

// Use the standard namespace
using namespace std;

//external global variables
extern int gRowIndex;
extern int gColIndex;
extern int baseOrderLg;
extern int baseOrder;
extern int baseOrder2;
extern int baseSize;
extern int baseSize2;

//-------------- Helpers -------------------

//Getting the sub-components of a given node
#define NW(quad) (	((quadNode*)(quad))->NW 	)
#define SW(quad) (	((quadNode*)(quad))->SW 	)
#define NE(quad) (	((quadNode*)(quad))->NE 	)
#define SE(quad) (	((quadNode*)(quad))->SE 	)

//2^n and 4^n
#define powerOf2(n) (1<<n)
#define powerOf4(n) (powerOf2(n*2))

/*
int expandSide(int side)
  expand side to the next multiple of baseOrder.
  param side: side to expand
  return: expanded side
*/
inline int expandSide(int side)
{
	return ((side+baseOrder-1)/baseOrder)*baseOrder;
}
inline int expandNextSide(int side)
{
	return ((side+baseOrder)/baseOrder)*baseOrder;
}

/*  !!! change to ceilLog4 ???
int nextLog4(int x)
  get the ceiling of the log base4 of a given number.
  param x: number to compute nextLog4 on.
  return: computed value
*/
inline int nextLog4(int x)
{
  int count = 0;
  x--;
  while(x>0){
    x = x>>2;
    count++;
  }
  return count;
}

/*
int isNullArray(dataType* arr, int arrSize)
  Does the given array of size arrSize contains
    nothing but zeroes?
  params
    arr: array pointer
    arrSize: size of array
  return: 1 if so, 0 if otherwise
*/
inline int isNullArray(dataType* arr, int arrSize)
{
	int i;
  	for(i=0; (i<arrSize) && ( arr[i] == 0 ); i++);
  	return (i == arrSize);
}

/*
void printArray(dataType* arr, int size)
  Print the given array of size arrSize
  params
    arr: array pointer
    arrSize: size of array
*/
inline void printArray(dataType* arr, int size)
{
	for(int i=0; i<size; i++)
		printf("   %lf\n",arr[i]);
}

/*
void printTabs(int level)
  Print 2 spaces per level
  used for tree printing
*/
inline void printTabs(int level)
{
	for(int i=0; i<level; i++)
	  printf("  ");
}

#endif
//////////////////////////////////////////////////////////////////////////////


