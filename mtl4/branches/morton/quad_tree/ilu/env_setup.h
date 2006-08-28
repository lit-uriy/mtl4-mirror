
/* 
  matrix environment setup for incomplete LU
    Global variables are defined in sysSolver.h
*/

#ifndef ENV_SETUP_H
#define ENV_SETUP_H

#include "matrix.h"


typedef struct f{
	char* name;
	char* description;
} flag;			// use for flags and their descriptions.
				    // --> helps when printing help.

//flags and brief descriptions - necessary for printing Help
flag flags[] = {
	{BASE_ORDER_LG,	"Base Order Log. (default is 3)."},
	{DAMP_FACT,	"Damping Factor for iterative solver."},
	{PRINT_INPUT,	"print out input and Specify format. (default: unset)"},
	{PRINT_OUTPUT,	"print out rhs w/ format= no_format. (default: unset)"},
	{PRINT_ALL,		"print out input and rhs. (default: unset)"},
	};
const int nFlags = 5;

//formats and brief descriptions - necessary for printing Help
flag formats[] = {	{ROW_MAJ_Z,		"Row-Major Order: zero values included"},
					{TREE_Z,		"Tree rep.: zero values included"},
					{HB,			"Harwell-Boeing Format - unsymmetric"},
					{HBS,			"Harwell-Boeing Format - symmetric"},
					{NO_FORMAT,		"unspecified format"},
					};
const int nFormats = 5;

//printing
int printInput = 0;			//print input?
int printOutput = 0;			//print rhs?
char printINformat[MAX_FLAG_SIZE];	//format used to print input matrices

//global vars
Matrix mat;
Memory<quadNode> quadStock(1);
Memory<dataType> bBlockStock;
int gRowIndex = 0;
int gColIndex = 0;
int alloc_step;
int baseOrderLg = 3;
int baseOrder;
int baseOrder2;
int baseSize;
int baseSize2;
double OMEGA = 1 ;
int numb_bblocks = 0;

//Function prototypes
void env_init(int nargc, char** nargv);
void env_kill();
void getUserArgs(int nargc, char** nargv);
void setFlag(char* flagName, char *value);
void checkFormat(char* fmt);
void printHelp();
void buildTree(char* fmt);
void printMatrix(Both node, char* printfmt);

#endif // ENV_SETUP_H

//////////////////////////////////////////////////////////////////////////////
