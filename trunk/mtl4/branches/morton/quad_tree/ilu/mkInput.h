/* file mkInput.h */

#ifndef MKINPUT_H
#define MKINPUT_H

#include <fstream>
#include <ctime>
# include <stdlib.h>
#include "defs.h"
#include "iohb.h"

#define FACT	1

int getMatInfo_HB(char* fileName, int *rows, int *cols, int *nRhs);
int getInput_HB ( char *fileName );
int getRowsAndCols_RM(char* fileName, int *rows, int *cols);
int getInput_RM ( char *fileName);


#endif

