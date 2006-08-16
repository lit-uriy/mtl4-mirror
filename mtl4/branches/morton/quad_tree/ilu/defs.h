/*****************************************************************************
  file: defs.h
  -------------
  Defines types and constants - and set flags - critical for the design and
    application of the sparse quadtree representation.

  Revised on: 07/25/06

  D. S. Wise
  Larisse D. Voufo
*****************************************************************************/
#ifndef DEFS_H
#define DEFS_H

using namespace std;

//___________ Type definitions ________________

typedef char decoration;
typedef double dataType;
typedef dataType* baseBlock;
typedef unsigned long int indexType;

// A quadNode contains points to 4 other nodes, representing
//  its northwest, southwest, northeast, and southeast components
union bt;
typedef struct {
	union bt* NW;  //northwest
	union bt* SW;  //southwest
	union bt* NE;  //northeast
	union bt* SE;  //southeast
} quadNode;

//A node of type Both can either be a quadNode or a baseBlock
typedef union bt {
	baseBlock bBlk;
	quadNode* quad;
}* Both;

typedef struct dc {
	int decor;
	struct dc* NW;
	struct dc* SE;
}* decorNode;


//___________ Flags ___________________________

#define DENSE_DIAG	1	//impose the condition that no diag element is nil? 1:0
					 		//*** not setting DENSE_DIAG does not help!!!
#define MTN_BBLK	0	// morton-ordered baseBlock?

//___________ Constants ________________

//User arguments' flags
#define HELP_OP       "H"     //print Help
#define BASE_ORDER_LG "-bol"  //base Order Log
#define DAMP_FACT     "-df"   //set the damping factor
#define PRINT_INPUT   "-pi"	  //print input matrices
#define PRINT_OUTPUT  "-po"	  //print output matrices
#define PRINT_ALL     "-pa"   //print all matrices: input and output

//Formats for printing output
#define ROW_MAJ_Z   "rmz"   //row-major order with zero values included
#define ROW_MAJ_NZ	"rmn"	 //row-major order - non-zero values only
									           //triplets of row index, col index, and value.
#define HB          "hb"	//harwell-Boeing Format - unsymmetric
#define HBS         "hbs"	//harwell-Boeing Format - symmetric
#define TREE_Z      "tz"	//Tree rep. with zero values included
#define NO_FORMAT   "nf"	//Do Not print the result check

//Other constants
#define NIL		0          //for NULL pointers
#define DECOR_SET 	'1'  //set a decoration
#define DECOR_UNSET	'0'  //unset a decoration
#define MTN_START   0    //starting morton index for a block
#define LEVEL_START 0    //starting level
#define SMALL_ERROR 0.0000000001 //Small error tolerance when comparing results

#endif
//////////////////////////////////////////////////////////////////////////////

